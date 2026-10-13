"""Scenes of the demo video: on-screen caption (title, sub) + voiceover text. Shared by tts.py and record_demo.py."""

LIVE_PROMPT = "a delicious hamburger"
LIVE_SEED = 0

SCENES = [
    dict(id="intro",
         title="Interactive Diffusion-Based 3D Generation Studio",
         sub="text → frozen 2D diffusion guidance → optimised 3D Gaussians → real-time novel views · running live on a local NVIDIA A100 (GPU 4)",
         say="This is my Interactive Diffusion-Based 3D Generation Studio for CS 549. It turns a text prompt into an "
             "explorable 3D object, demonstrating two graphics pillars: procedural generation and rendering. Everything runs live on a local NVIDIA A100."),
    dict(id="prompt",
         title=f"Prompt: “{LIVE_PROMPT}”",
         sub="POST /api/generate → official DreamGaussian code (commit 69a3c70) on the GPU",
         say="I type a new prompt, a delicious hamburger, and press generate. The server launches the official "
             "DreamGaussian code on the GPU."),
    dict(id="loading",
         title="Loading frozen Stable Diffusion 2.1 · initialising 5,000 random Gaussians",
         sub="the diffusion model is never trained — it only scores rendered images",
         say="The frozen Stable Diffusion model is loaded, and optimisation starts from a random blob of five thousand "
             "Gaussians."),
    dict(id="train1",
         title="Score Distillation: render → add noise → U-Net predicts noise → gradient into the Gaussians",
         sub="random camera each step · differentiable CUDA Gaussian rasterizer · CFG 100 · timestep annealed 980 → 20",
         say="Each step renders a random view with the differentiable CUDA rasterizer. Noise is added, the diffusion "
             "U-Net predicts it, and the score-distillation gradient flows back through the renderer into the Gaussians."),
    dict(id="train2",
         title="Monitor: non-zero gradients reach position, colour, opacity, scale and rotation",
         sub="densify/prune every 50 steps · training resolution 128 → 256 → 512",
         say="The monitor shows the guidance really flows: every Gaussian parameter receives gradient, while "
             "densification grows the scene to hundreds of thousands of Gaussians."),
    dict(id="train3",
         title="Snapshots rendered by the CUDA rasterizer during optimisation",
         sub="front · side · back · side — blob → shape → detail",
         say="These snapshots come from the same rasterizer during training. The blob becomes a hamburger, and five "
             "hundred steps take under a minute."),
    dict(id="export",
         title="Saving Gaussian PLY + extracting textured mesh   ⏩ fast-forward 8×",
         sub="upstream save_model('model') and save_model('geo+tex'), then 8 novel views + turntable",
         say="Training is done. The Gaussians are saved, and a textured mesh is extracted, shown here at eight times speed."),
    dict(id="viewer",
         title="Real-time novel views: WebGL2 Gaussian splatting viewer",
         sub="per-Gaussian 3D covariance → EWA projection → depth sort (Web Worker) → alpha blending",
         say="The new model loads into my WebGL 2 splatting viewer. Each Gaussian is projected with its covariance, "
             "depth sorted, and blended, so I can orbit and zoom in real time."),
    dict(id="primitives",
         title="Shrinking each splat reveals the explicit 3D Gaussian primitives",
         sub="scrubber replays the optimisation snapshots: step 0 → final",
         say="Shrinking the splats reveals the explicit primitives the diffusion guidance placed in space, and the "
             "scrubber replays the optimisation."),
    dict(id="stress1",
         title="Stress test: CUDA Gaussian rasterizer on the A100",
         sub="fixed 360° orbit · warm-up excluded · synced timing · sweep resolution, then clone Gaussians until OOM",
         say="Now the stress test. The CUDA rasterizer renders a fixed orbit while I sweep the resolution, then clone "
             "Gaussians until memory runs out."),
    dict(id="stress2",
         title="{STRESS_TITLE}",
         sub="{STRESS_SUB}",
         say="The measured numbers fill in live. Frame rate falls as resolution and Gaussian count grow, "
             "and the sweep ends where the GPU runs out of memory. That is the limit of this renderer."),
    dict(id="stress3",
         title="Browser viewer stress: model replicated ×1 … ×16 (≈4.6M Gaussians)",
         sub="requestAnimationFrame FPS + GPU timer query + worker sort time",
         say="The browser viewer is stressed the same way, replicating the model up to sixteen times, almost five million Gaussians, all sorted and blended in the browser."),
    dict(id="outro",
         title="Plan vs. reality: SDS + NeRF (A2 plan) → SDS + explicit 3D Gaussians (DreamGaussian)",
         sub="same diffusion-guided loop · next: geometry quality, seed/guidance comparisons, mesh refinement, viewer polish",
         say="Plan versus reality: Assignment 2 planned a NeRF. I kept the same diffusion-guided loop, but switched to "
             "explicit 3D Gaussians for speed. Next come better geometry, guidance comparisons, and mesh refinement."),
]


def fill_stress(csv_path):
    """Fill the stress-result scene from the measured CSV (numbers are never typed by hand)."""
    import csv
    rows = list(csv.DictReader(open(csv_path)))
    res = {int(r["resolution"]): float(r["fps"]) for r in rows if r["setting"] == "resolution" and float(r["fps"]) > 0}
    cnt = [(int(r["gaussians"]), float(r["fps"]), float(r["peak_vram_mb"]), r["notes"]) for r in rows if r["setting"] == "gaussians"]
    ok = [c for c in cnt if c[1] > 0]
    oom = [c for c in cnt if "OOM" in c[3]]
    big = ok[-1]
    title = (f"Measured: {res[512]:.0f} FPS @512² · {res[2048]:.0f} FPS @2048² · {res[4096]:.0f} FPS @4096² "
             f"· {big[1]:.1f} FPS with {big[0] / 1e6:.0f}M Gaussians")
    sub = (f"peak VRAM {big[2] / 1024:.1f} GB at {big[0] / 1e6:.0f}M Gaussians"
           + (f" · CUDA out-of-memory at {oom[0][0] / 1e6:.0f}M Gaussians (failure point)" if oom else ""))
    for s in SCENES:
        if s["id"] == "stress2":
            s.update(title=title, sub=sub)
    return SCENES
