"""Run upstream DreamGaussian text-to-3D (stage 1, Gaussian splatting + SDS) with evidence logging.

The optimisation itself is upstream code, unmodified (third_party/dreamgaussian/main.py::GUI.train_step).
This wrapper only adds instrumentation around it:
  * per-step SDS loss, diffusion timestep and gradient norms that reach each Gaussian parameter group
  * periodic snapshot renders (front view + 4-view strip) so intermediate vs. final can be compared
  * progress.json (polled by the web studio), metrics.jsonl, summary.json, peak VRAM, wall-clock time
  * after training: Gaussian PLY, textured mesh (upstream geo+tex export), novel views and a turntable MP4

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/train_logged.py --prompt "a small red mushroom house" --name mushroom_house
"""
import argparse
import functools
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DG_ROOT = os.path.join(ROOT, "third_party", "dreamgaussian")
sys.path.insert(0, DG_ROOT)

import numpy as np
import torch
import imageio
from omegaconf import OmegaConf

import guidance.sd_utils as sd_utils
from main import GUI
from cam_utils import orbit_camera
from gs_renderer import MiniCam

# stabilityai/stable-diffusion-2-1-base is no longer downloadable (HTTP 401 as of 2026-10);
# this is a byte-identical community mirror of the same SD 2.1-base diffusers weights.
SD_KEY = os.environ.get("DG_SD_KEY", "Manojb/stable-diffusion-2-1-base")


def now():
    return time.perf_counter()


class LoggedTrainer(GUI):
    def __init__(self, opt, args):
        self.args = args
        self.out_dir = args.out
        os.makedirs(os.path.join(self.out_dir, "progress"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "views"), exist_ok=True)
        self.metrics_f = open(os.path.join(self.out_dir, "metrics.jsonl"), "w")
        self.last = {}
        super().__init__(opt)

    # ---------------------------------------------------------------- helpers
    def write_progress(self, status, **extra):
        d = dict(
            status=status,
            prompt=self.prompt,
            seed=self.args.seed,
            step=self.step,
            iters=self.opt.iters,
            elapsed_s=round(now() - self.t0, 2) if hasattr(self, "t0") else 0,
            num_gaussians=int(self.renderer.gaussians.get_xyz.shape[0]),
            peak_vram_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1),
            **self.last,
            **extra,
        )
        tmp = os.path.join(self.out_dir, "progress.json.tmp")
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, os.path.join(self.out_dir, "progress.json"))
        return d

    @torch.no_grad()
    def render_view(self, ver, hor, res, bg=(1, 1, 1)):
        pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius)
        cam = MiniCam(pose, res, res, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
        out = self.renderer.render(cam, bg_color=torch.tensor(bg, dtype=torch.float32, device="cuda"))
        img = out["image"].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        return (img * 255).astype(np.uint8)

    def snapshot(self, tag):
        strip = np.concatenate([self.render_view(-15, h, 256) for h in (0, 90, 180, 270)], axis=1)
        imageio.imwrite(os.path.join(self.out_dir, "progress", f"{tag}.png"), strip)
        imageio.imwrite(os.path.join(self.out_dir, "progress", "latest.png"), strip)

    # ------------------------------------------------------- instrumentation
    def instrument(self):
        sd = self.guidance_sd
        orig_train_step = sd.train_step

        @functools.wraps(orig_train_step)
        def train_step(pred_rgb, step_ratio=None, **kw):
            kw.setdefault("guidance_scale", self.args.guidance_scale)
            loss = orig_train_step(pred_rgb, step_ratio=step_ratio, **kw)
            # upstream anneals t deterministically from step_ratio; recompute it for the log
            if step_ratio is not None:
                t = float(np.clip(np.round((1 - step_ratio) * sd.num_train_timesteps), sd.min_step, sd.max_step))
            else:
                t = float("nan")
            self.last["sds_loss"] = float(loss.detach())
            self.last["timestep"] = t
            self.last["render_res"] = int(pred_rgb.shape[-1])
            return loss

        sd.train_step = train_step

        gs = self.renderer.gaussians
        orig_step = self.optimizer.step

        def opt_step(*a, **kw):
            # gradient norms that the SDS loss back-propagated through the rasterizer into each parameter group
            g = {}
            for name in ("_xyz", "_features_dc", "_opacity", "_scaling", "_rotation"):
                p = getattr(gs, name)
                g[name.strip("_")] = 0.0 if p.grad is None else float(p.grad.norm())
            self.last["grad_norm"] = g
            return orig_step(*a, **kw)

        self.optimizer.step = opt_step

    # ------------------------------------------------------------------ train
    def train(self, iters=500):
        a = self.args
        # seed RNGs only; upstream's GUI.seed_everything() would also flip cudnn.benchmark/deterministic,
        # which the upstream headless path never does (and benchmark autotuning inflates peak VRAM)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed(a.seed)
        torch.cuda.reset_peak_memory_stats()
        self.t0 = now()
        self.write_progress("loading_diffusion_model")

        t_load = now()
        self.prepare_train()
        self.instrument()
        load_s = now() - t_load
        print(f"[EVIDENCE] diffusion guidance model loaded in {load_s:.1f}s ({SD_KEY})", flush=True)

        self.snapshot("step_0000")
        self.write_progress("training", load_s=round(load_s, 1))

        t_train = now()
        for i in range(iters):
            self.train_step()
            rec = dict(step=self.step, t_s=round(now() - t_train, 3),
                       num_gaussians=int(self.renderer.gaussians.get_xyz.shape[0]), **self.last)
            self.metrics_f.write(json.dumps(rec) + "\n")
            self.metrics_f.flush()  # the studio charts read this file live
            if self.step % a.log_every == 0 or self.step == iters:
                gn = self.last["grad_norm"]
                print(f"[step {self.step:4d}/{iters}] sds_loss={self.last['sds_loss']:.2f} t={self.last['timestep']:.0f} "
                      f"res={self.last['render_res']} N={rec['num_gaussians']} "
                      f"|grad| xyz={gn['xyz']:.3e} color={gn['features_dc']:.3e} opac={gn['opacity']:.3e} "
                      f"scale={gn['scaling']:.3e} rot={gn['rotation']:.3e} "
                      f"vram_peak={torch.cuda.max_memory_allocated()/2**20:.0f}MB", flush=True)
            if self.step % a.snap_every == 0:
                self.snapshot(f"step_{self.step:04d}")
                self.write_progress("training")
        torch.cuda.synchronize()
        train_s = now() - t_train
        self.metrics_f.close()

        # upstream: final prune, then save Gaussians (+ mesh)
        self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
        self.snapshot("final")
        self.write_progress("exporting", train_s=round(train_s, 1))
        t_exp = now()
        self.save_model(mode="model")
        if a.mesh:
            self.save_model(mode="geo+tex")
        export_s = now() - t_exp

        # novel views + turntable from the optimised Gaussians
        t_vis = now()
        for h in range(0, 360, 45):
            imageio.imwrite(os.path.join(self.out_dir, "views", f"view_{h:03d}.png"), self.render_view(-15, h, 512))
        frames = [self.render_view(-15 + 10 * math.sin(2 * math.pi * k / a.turntable_frames), 360 * k / a.turntable_frames, 512)
                  for k in range(a.turntable_frames)]
        imageio.mimwrite(os.path.join(self.out_dir, "turntable.mp4"), frames, fps=30, quality=8, macro_block_size=1)
        imageio.mimwrite(os.path.join(self.out_dir, "turntable.gif"), frames[::3], duration=0.1, loop=0)
        vis_s = now() - t_vis

        summary = dict(
            prompt=self.prompt, seed=a.seed, iters=iters, guidance_scale=a.guidance_scale,
            sd_model=SD_KEY, gpu=torch.cuda.get_device_name(0),
            num_gaussians_final=int(self.renderer.gaussians.get_xyz.shape[0]),
            load_model_s=round(load_s, 1), train_s=round(train_s, 1), export_s=round(export_s, 1),
            render_views_s=round(vis_s, 1), total_s=round(now() - self.t0, 1),
            peak_vram_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1),
            ply=f"{self.opt.save_path}_model.ply",
            mesh=f"{self.opt.save_path}_mesh.obj" if a.mesh else None,
            config_overrides=self.overrides,
        )
        with open(os.path.join(self.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        self.write_progress("done", **{k: summary[k] for k in ("train_s", "total_s", "num_gaussians_final")})
        print("[EVIDENCE] summary " + json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--guidance_scale", type=float, default=100.0)
    p.add_argument("--out", default=None)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--snap_every", type=int, default=25)
    p.add_argument("--turntable_frames", type=int, default=120)
    p.add_argument("--no_mesh", dest="mesh", action="store_false")
    a = p.parse_args()
    a.out = a.out or os.path.join(ROOT, "results", a.name)
    os.makedirs(a.out, exist_ok=True)

    # patch only the model key passed to upstream's StableDiffusion guidance class
    sd_utils.StableDiffusion = functools.partial(sd_utils.StableDiffusion, hf_key=SD_KEY)

    base = OmegaConf.load(os.path.join(DG_ROOT, "configs", "text.yaml"))
    overrides = dict(prompt=a.prompt, negative_prompt=a.negative_prompt, save_path=a.name,
                     outdir=a.out, iters=a.iters, gui=False, force_cuda_rast=True)
    opt = OmegaConf.merge(base, OmegaConf.create(overrides))
    OmegaConf.save(opt, os.path.join(a.out, "config_resolved.yaml"))

    print(f"[EVIDENCE] torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"[EVIDENCE] overrides={overrides} guidance_scale={a.guidance_scale} seed={a.seed}", flush=True)

    trainer = LoggedTrainer(opt, a)
    trainer.overrides = dict(overrides, guidance_scale=a.guidance_scale, seed=a.seed)
    try:
        trainer.train(opt.iters)
    except Exception as e:
        trainer.write_progress("failed", error=repr(e))
        raise


if __name__ == "__main__":
    main()
