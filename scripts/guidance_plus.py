"""Alternative score-distillation losses on top of upstream DreamGaussian's frozen Stable Diffusion guidance.

All methods reuse the same SD 2.1-base UNet/VAE/text encoder held by guidance.sd_utils.StableDiffusion (no new
weights, no training of the diffusion model) and return a surrogate loss whose gradient w.r.t. the rendered image's
latents equals the method's update direction (same trick as upstream SDS: 0.5*||z - sg(z - g)||^2).

  csd  - Classifier Score Distillation (Yu et al., ICLR 2024): g = w(t) * s * (eps(x_t,y) - eps(x_t,0)).
         Drops SDS's (eps_uncond - noise) residual; sharper textures, same cost as SDS.
  nfsd - Noise-Free Score Distillation (Katzir et al., ICLR 2024): g = w(t) * (delta_D + s * delta_C),
         delta_C = eps(x_t,y) - eps(x_t,0), delta_D = eps(x_t,0) - eps(x_t,neg_quality) for t >= 200 else eps(x_t,0).
         Works at normal CFG (7.5) -> far less over-saturation. 3 UNet passes.
  ism  - Interval Score Matching (Liang et al., LucidDreamer, CVPR 2024): DDIM-invert the clean latent to s = t - dt
         with the unconditional model, take one more inversion step to t, and use
         g = w(t) * (eps_cfg(x_t, t, y) - eps(x_s, s, 0)).
         Deterministic, low-variance target instead of random noise; reported to beat VSD (ProlificDreamer)
         in quality at a fraction of the cost. (inv_steps + 3) UNet passes.

Gradients are rescaled by grad_scale (default 100 / cfg) so their magnitude roughly matches upstream SDS at CFG 100;
upstream densification thresholds act on screen-space gradient magnitudes and were tuned for that scale.
"""
import numpy as np
import torch
import torch.nn.functional as F

NFSD_NEG = "unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, dark, low-resolution, gloomy"


def _dir(h):
    return "front" if abs(h) < 60 else ("side" if abs(h) < 120 else "back")


class GuidancePlus:
    def __init__(self, sd, method, cfg, grad_scale=None, view_prompts=True, ism_dt=50, ism_inv_steps=3, get_hors=None):
        assert method in ("csd", "nfsd", "ism")
        self.sd, self.method, self.cfg = sd, method, cfg
        self.grad_scale = grad_scale if grad_scale is not None else 100.0 / cfg
        self.view_prompts = view_prompts
        self.ism_dt, self.ism_inv_steps = ism_dt, ism_inv_steps
        self.get_hors = get_hors or (lambda n: None)
        self.alphas = sd.alphas.to(torch.float32)  # alphas_cumprod
        with torch.no_grad():
            self.neg_quality = sd.encode_text([NFSD_NEG]) if method == "nfsd" else None
        self.last_t = float("nan")

    # ------------------------------------------------------------- helpers
    def _cond(self, B):
        sd = self.sd
        hors = self.get_hors(B) if self.view_prompts else None
        if hors is not None and all(d in sd.embeddings for d in ("front", "side", "back")):
            return torch.cat([sd.embeddings[_dir(h)] for h in hors])
        return sd.embeddings["pos"].expand(B, -1, -1)

    def _unet(self, x, t, emb):
        return self.sd.unet(x.to(self.sd.dtype), t, encoder_hidden_states=emb.to(self.sd.dtype)).sample.float()

    def _timestep(self, B, step_ratio):
        sd = self.sd
        if step_ratio is not None:  # same deterministic annealing as upstream SDS
            t = np.round((1 - step_ratio) * sd.num_train_timesteps).clip(sd.min_step, sd.max_step)
            return torch.full((B,), int(t), dtype=torch.long, device=sd.device)
        return torch.randint(sd.min_step, sd.max_step + 1, (B,), dtype=torch.long, device=sd.device)

    def _a(self, t):  # alpha_bar as [B,1,1,1]; t < 0 means the clean latent
        a = torch.where(t >= 0, self.alphas[t.clamp(min=0)], torch.ones_like(t, dtype=torch.float32))
        return a.view(-1, 1, 1, 1)

    def _ddim(self, x, eps, t_from, t_to):
        a0, a1 = self._a(t_from), self._a(t_to)
        x0 = (x - (1 - a0).sqrt() * eps) / a0.sqrt()
        return a1.sqrt() * x0 + (1 - a1).sqrt() * eps

    # ---------------------------------------------------------------- loss
    def train_step(self, pred_rgb, step_ratio=None, guidance_scale=None, **kw):
        sd = self.sd
        B = pred_rgb.shape[0]
        cfg = self.cfg
        latents = sd.encode_imgs(F.interpolate(pred_rgb.to(sd.dtype), (512, 512), mode="bilinear", align_corners=False)).float()
        t = self._timestep(B, step_ratio)
        self.last_t = float(t[0])
        w = (1 - self._a(t))
        cond = self._cond(B)
        uncond = sd.embeddings["neg"].expand(B, -1, -1)

        with torch.no_grad():
            if self.method in ("csd", "nfsd"):
                noise = torch.randn_like(latents)
                x_t = sd.scheduler.add_noise(latents, noise, t)
                embs = [cond, uncond] + ([self.neg_quality.expand(B, -1, -1)] if self.method == "nfsd" else [])
                eps = self._unet(torch.cat([x_t] * len(embs)), torch.cat([t] * len(embs)), torch.cat(embs)).chunk(len(embs))
                delta_c = eps[0] - eps[1]
                if self.method == "csd":
                    grad = w * cfg * delta_c
                else:
                    small_t = (t < 200).view(-1, 1, 1, 1)
                    delta_d = torch.where(small_t, eps[1], eps[1] - eps[2])
                    grad = w * (delta_d + cfg * delta_c)
            else:  # ism
                s = (t - self.ism_dt).clamp(min=0)
                x, cur = latents, torch.full_like(t, -1)
                for i in range(1, self.ism_inv_steps + 1):  # DDIM inversion 0 -> s with the unconditional model
                    nxt = (s.float() * i / self.ism_inv_steps).round().long()
                    eps = self._unet(x, cur.clamp(min=0), uncond)
                    x, cur = self._ddim(x, eps, cur, nxt), nxt
                eps_s = self._unet(x, s, uncond)                       # eps(x_s, s, 0)
                x_t = self._ddim(x, eps_s, s, t)                         # one inversion step s -> t
                e = self._unet(torch.cat([x_t, x_t]), torch.cat([t, t]), torch.cat([cond, uncond])).chunk(2)
                eps_t = e[1] + cfg * (e[0] - e[1])                       # eps_cfg(x_t, t, y)
                grad = w * (eps_t - eps_s)
            grad = torch.nan_to_num(grad * self.grad_scale)

        target = (latents - grad).detach()
        return 0.5 * F.mse_loss(latents, target, reduction="sum") / B
