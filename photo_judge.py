"""CLIP judge for player photos (Beta "photo shoot").

Same judge as the end-of-job check in scripts/train_logged.py (ViT-L/14, 20-way zero-shot over configs/challenge.json,
softmax(100*cos) averaged over images), but it scores the images the player rendered in the browser: their camera
angle, lighting and backdrop. It runs on the CPU so it never takes a GPU that someone else might want.
"""
import base64
import io
import json
import os
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
CLIP_KEY = "openai/clip-vit-large-patch14"
MAX_BYTES = 3_000_000
_lock = threading.Lock()
_state = {}


def _load():
    if "model" in _state:
        return _state
    os.environ.setdefault("HF_HUB_OFFLINE", "1")        # weights are in the local cache; never wait on the Hub
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import CLIPModel, CLIPProcessor
    torch.set_num_threads(int(os.environ.get("JUDGE_THREADS", 16)))
    labels = [t["label"] for t in json.load(open(os.path.join(ROOT, "configs", "challenge.json")))["targets"]]
    model = CLIPModel.from_pretrained(CLIP_KEY).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_KEY)
    with torch.no_grad():
        t = proc(text=[f"a 3D render of {l}" for l in labels], return_tensors="pt", padding=True)
        tx = model.get_text_features(**t)
    _state.update(torch=torch, model=model, proc=proc, labels=labels, tx=tx / tx.norm(dim=-1, keepdim=True))
    return _state


def decode(data_url):
    """data:image/png;base64,... -> RGB PIL image (raises ValueError on anything else)."""
    from PIL import Image
    if not isinstance(data_url, str) or not data_url.startswith(("data:image/png;base64,", "data:image/jpeg;base64,")):
        raise ValueError("photos must be PNG or JPEG data URLs")
    raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    if len(raw) > MAX_BYTES:
        raise ValueError("photo too large")
    im = Image.open(io.BytesIO(raw))
    if im.width * im.height > 2048 * 2048:
        raise ValueError("photo too large")
    return im.convert("RGB")


def judge(images, prompt="", target=None):
    t0 = time.time()
    with _lock:
        s = _load()
        torch = s["torch"]
        with torch.no_grad():
            x = s["proc"](images=images, return_tensors="pt")
            im = s["model"].get_image_features(**x)
            im = im / im.norm(dim=-1, keepdim=True)
            sim = im @ s["tx"].T
            prob = (100 * sim).softmax(-1).mean(0)
            pc = None
            if prompt:
                t = s["proc"](text=[prompt[:300]], return_tensors="pt", padding=True, truncation=True)
                pt = s["model"].get_text_features(**t)
                pc = float((im @ (pt / pt.norm(dim=-1, keepdim=True)).T).mean() * 100)
    labels = s["labels"]
    order = prob.argsort(descending=True).tolist()
    res = dict(top=[dict(label=labels[i], p=round(float(prob[i]), 3)) for i in order[:5]],
               prompt_clip=round(pc, 2) if pc is not None else None, n=len(images))
    if target in labels:
        ti = labels.index(target)
        res.update(target=target, target_p=round(float(prob[ti]), 3), target_rank=order.index(ti) + 1, win=order[0] == ti)
    res["judge_s"] = round(time.time() - t0, 2)
    return res


if __name__ == "__main__":  # quick check: python photo_judge.py results/corgi/views/view_000.png "a corgi dog"
    import sys
    from PIL import Image
    print(judge([Image.open(p).convert("RGB") for p in sys.argv[1:-1]], target=sys.argv[-1]))
