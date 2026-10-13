"""Scenes of demo video v2 (final studio): caption (title, sub) + voiceover. Used by tts.py (SCRIPT=2) and record_demo2.py."""

LIVE_PROMPT = "a cute owl plush toy"
STOP_PROMPT = "a blue ceramic teapot"

SCENES = [
    dict(id="intro",
         title="Interactive Diffusion-Based 3D Generation Studio",
         sub="text → frozen diffusion guidance → optimised 3D Gaussians → real-time rendering · public web app on an A100 server",
         say="This is my Interactive Diffusion-Based 3D Generation Studio for CS 549. It turns a text prompt into a 3D object, "
             "and it now runs as a public web app on our lab's A100 server."),
    dict(id="controls",
         title="Pick a guidance method · see free GPUs and the ETA before submitting",
         sub="ETA = measured seconds per step × steps + overhead, simulated over the current queue",
         say="Visitors choose a guidance method. Before submitting, the page shows how many GPUs are free, and an estimated "
             "finish time computed from measured timings."),
    dict(id="submit",
         title=f"Prompt: “{LIVE_PROMPT}” · MVDream guidance · 500 steps",
         sub="the job is sent to a GPU that has no other process on it",
         say="I'll generate a cute owl plush toy with MVDream. The server sends the job to a GPU that nobody else is using."),
    dict(id="loading",
         title="Loading the frozen MVDream multi-view diffusion model   ⏩ fast-forward",
         sub="5k random Gaussians are initialised; the diffusion model is never trained",
         say="While the frozen diffusion model loads, five thousand random Gaussians are created."),
    dict(id="train1",
         title="Multi-view score distillation: 4 views, 90° apart, scored together",
         sub="differentiable CUDA Gaussian rasterizer → MVDream noise prediction → gradient into every Gaussian",
         say="Each step renders four views ninety degrees apart. The multi-view diffusion model scores them together, "
             "so the object stays consistent from every side."),
    dict(id="train2",
         title="Live monitor: gradients reach position, colour, opacity, scale and rotation",
         sub="densification grows the scene; the job box counts down to the predicted finish time",
         say="The monitor shows gradients reaching every Gaussian parameter, and the job box counts down to its predicted finish."),
    dict(id="ff_train",
         title="Optimisation continues   ⏩ fast-forward 6×",
         sub="500 steps ≈ 90 s on one A100",
         say="Five hundred steps take about a minute and a half."),
    dict(id="viewer",
         title="Real-time novel views in the WebGL2 Gaussian splatting viewer",
         sub="EWA projection of each 3D covariance · depth sort in a Web Worker · alpha blending",
         say="The finished owl loads into my WebGL splatting viewer, and I can orbit and zoom in real time."),
    dict(id="stop",
         title="One job per session at a time · Stop releases the GPU immediately",
         sub="queued jobs show their position and ETA; a stopped job's GPU goes to the next person in line",
         say="Each session can run one job at a time. Stop cancels it and frees the GPU right away for the next person in the queue."),
    dict(id="gallery",
         title="Shared gallery: every finished model can be rendered without re-running",
         sub="curated examples + community models",
         say="Every finished model joins the shared gallery, so anyone can render it without running the optimisation again."),
    dict(id="algo",
         title="Improving DreamGaussian: SDS vs CSD vs NFSD vs ISM vs MVDream (measured)",
         sub="same 5 prompts, seed and steps · CLIP score, time, VRAM · front | back views",
         say="To improve quality, I implemented C S D, N F S D and Interval Score Matching, enabled MVDream, and compared them "
             "on the same prompts. MVDream removes the two-faced artifacts and gives the sharpest textures, so it is the default."),
    dict(id="steps",
         title="Step sweep: 500 steps is the sweet spot",
         sub="300 softer · 500 sharpest · 800 and 1200 over-saturate",
         say="A step sweep showed that five hundred steps looks best. More steps over-saturate the colours."),
    dict(id="stress",
         title="Stress test: CUDA rasterizer limits (measured) + browser viewer pushed live ×1 → ×16",
         sub="CUDA: 368 FPS @512² → 16.8 FPS @4096², 5.5 FPS at 40M Gaussians, out-of-memory at 80M · browser: watch FPS fall as copies grow",
         say="For the stress test, the CUDA rasterizer measured about three hundred and seventy frames per second at five-twelve "
             "squared, falling until the GPU ran out of memory at eighty million Gaussians. Here the browser viewer is pushed live, "
             "from one to sixteen copies of the model."),
    dict(id="outro",
         title="Plan vs. reality: SDS + NeRF (A2) → diffusion-guided 3D Gaussians + MVDream",
         sub="next: combine MVDream with ISM targets, mesh refinement, polish the studio",
         say="Assignment 2 planned a NeRF. I kept the diffusion-guided loop, but moved to 3D Gaussians and multi-view guidance "
             "for speed and quality. Next come mesh refinement and polishing the studio."),
]
