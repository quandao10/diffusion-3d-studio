"""Compare score-distillation variants on the same prompts/seed/steps, then score them.

Runs scripts/train_logged.py for every (method, prompt) on process-free GPUs (one job per GPU, a GPU is used only when
nvidia-smi lists no process on it), then computes the CLIP score (ViT-L/14 cosine similarity x 100 between the prompt
and 8 novel views at 45 deg steps) and writes evidence/guidance_comparison.csv + docs_assets/guidance_grid.png.

usage: python scripts/compare_guidance.py [--methods sds,csd,nfsd,ism,mvdream] [--gpus 4,5,6,7] [--eval_only]
"""
import argparse, csv, json, os, queue, subprocess, sys, threading, time
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "guidance")
PROMPTS = ["a small red mushroom house", "a cute corgi puppy", "a blue ceramic teapot", "a delicious hamburger", "a ripe pineapple"]


def slug(s):
    return "".join(c if c.isalnum() else "_" for c in s.lower()).strip("_")[:40]


def gpu_has_process(i):
    xml = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True).stdout
    g = ET.fromstring(xml).findall("gpu")[i]
    return len(g.findall("processes/process_info")) > 0


SUFFIX = ""  # e.g. "_it1000" for step-count sweeps: runs go to experiments/guidance/<method><suffix>/


def run_all(methods, gpus, iters, seed):
    jobs = queue.Queue()
    for m in methods:
        for p in PROMPTS:
            d = os.path.join(OUT, m + SUFFIX, slug(p))
            if not os.path.exists(os.path.join(d, "summary.json")):
                jobs.put((m, p, d))
    print(f"[compare] {jobs.qsize()} runs on GPUs {gpus}", flush=True)

    def worker(g):
        while True:
            try:
                m, p, d = jobs.get_nowait()
            except queue.Empty:
                return
            while gpu_has_process(g):  # never share a GPU (fair timing, and never disturb other users)
                time.sleep(5)
            os.makedirs(d, exist_ok=True)
            cmd = [sys.executable, "scripts/train_logged.py", "--prompt", p, "--name", slug(p), "--guidance", m,
                   "--seed", str(seed), "--iters", str(iters), "--no_mesh", "--turntable_frames", "60", "--out", d]
            t = time.time()
            with open(os.path.join(d, "training.log"), "w") as log:
                rc = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)))
            print(f"[compare] GPU{g} {m:8s} {p:30s} rc={rc} {time.time() - t:.0f}s", flush=True)

    ts = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    [t.start() for t in ts]
    [t.join() for t in ts]


def evaluate(methods):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    rows = []
    for m in methods:
        for p in PROMPTS:
            d = os.path.join(OUT, m + SUFFIX, slug(p))
            s = json.load(open(os.path.join(d, "summary.json")))
            ims = [Image.open(os.path.join(d, "views", f"view_{h:03d}.png")).convert("RGB") for h in range(0, 360, 45)]
            with torch.no_grad():
                x = proc(text=[p], images=ims, return_tensors="pt", padding=True).to(dev)
                o = model(**x)
                ti = o.text_embeds / o.text_embeds.norm(dim=-1, keepdim=True)
                im = o.image_embeds / o.image_embeds.norm(dim=-1, keepdim=True)
                clip = float((im @ ti.T).mean() * 100)
            rows.append(dict(method=m + SUFFIX, iters=s["iters"], prompt=p, clip_score=round(clip, 2), train_s=s["train_s"], total_s=s["total_s"],
                             gaussians=s["num_gaussians_final"], peak_vram_mb=s["peak_vram_mb"], guidance_scale=s.get("guidance_scale")))
            print(rows[-1], flush=True)
    path = os.path.join(ROOT, "evidence", f"guidance_comparison{SUFFIX}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("[compare] per-method mean:")
    for m in methods:
        r = [x for x in rows if x["method"] == m + SUFFIX]
        print(f"  {m:8s} CLIP {sum(x['clip_score'] for x in r) / len(r):6.2f}   train {sum(x['train_s'] for x in r) / len(r):6.1f}s"
              f"   VRAM {max(x['peak_vram_mb'] for x in r) / 1024:.1f} GB")
    return rows


def grid(methods):
    from PIL import Image, ImageDraw, ImageFont
    S = 192
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except OSError:
        f = None
    W = Image.new("RGB", (110 + len(PROMPTS) * 2 * S, 30 + len(methods) * S), "white")
    d = ImageDraw.Draw(W)
    for j, p in enumerate(PROMPTS):
        d.text((110 + j * 2 * S + 6, 6), p[:34] + " (front | back)", fill=(20, 20, 20), font=f)
    for i, m in enumerate(methods):
        d.text((8, 30 + i * S + S // 2 - 8), (m + SUFFIX).upper(), fill=(20, 20, 20), font=f)
        for j, p in enumerate(PROMPTS):
            base = os.path.join(OUT, m + SUFFIX, slug(p), "views")
            for k, h in enumerate((0, 180)):
                W.paste(Image.open(os.path.join(base, f"view_{h:03d}.png")).resize((S, S)), (110 + (2 * j + k) * S, 30 + i * S))
    path = os.path.join(ROOT, "docs_assets", f"guidance_grid{SUFFIX}.png")
    W.save(path)
    print("[compare] wrote", path)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--methods", default="sds,csd,nfsd,ism,mvdream")
    a.add_argument("--gpus", default="4,5,6,7")
    a.add_argument("--iters", type=int, default=500)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--eval_only", action="store_true")
    a.add_argument("--suffix", default="")
    a = a.parse_args()
    SUFFIX = a.suffix
    methods = a.methods.split(",")
    if not a.eval_only:
        run_all(methods, [int(g) for g in a.gpus.split(",")], a.iters, a.seed)
    evaluate(methods)
    grid(methods)
