"""Accuracy of the Challenge-mode CLIP judge on every existing result whose prompt names one of the 20 targets.
Runs on CPU by default (CUDA_VISIBLE_DEVICES=""). Writes evidence/judge_eval.csv and prints top-1 accuracy."""
import csv, glob, json, os, sys
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(os.path.join(ROOT, "configs", "challenge.json")))["targets"]
labels = [t["label"] for t in cfg]
KEY = {"mushroom": "a mushroom", "corgi": "a corgi dog", "teapot": "a teapot", "hamburger": "a hamburger", "pineapple": "a pineapple",
       "owl": "an owl", "treasure chest": "a treasure chest", "cactus": "a cactus", "car": "a car"}
m = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
pr = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
with torch.no_grad():
    te = m.get_text_features(**pr(text=[f"a 3D render of {l}" for l in labels], return_tensors="pt", padding=True))
    te = te / te.norm(dim=-1, keepdim=True)
rows = []
for s in sorted(glob.glob(os.path.join(ROOT, "experiments/guidance/*/*/summary.json")) + sorted(glob.glob(os.path.join(ROOT, "results/*/summary.json")))
               + sorted(glob.glob(os.path.join(ROOT, "results/public/*/summary.json")))):
    d = os.path.dirname(s)
    prompt = json.load(open(s))["prompt"].lower()
    exp = next((v for k, v in KEY.items() if k in prompt), None)
    if exp is None:
        continue
    ims = [Image.open(f).convert("RGB") for f in sorted(glob.glob(os.path.join(d, "views", "view_*.png")))]
    with torch.no_grad():
        ie = m.get_image_features(**pr(images=ims, return_tensors="pt"))
        ie = ie / ie.norm(dim=-1, keepdim=True)
        p = (100 * ie @ te.T).softmax(-1).mean(0)
    order = p.argsort(descending=True).tolist()
    rows.append(dict(run=os.path.relpath(d, ROOT), prompt=prompt, expected=exp, top1=labels[order[0]], top1_p=round(float(p[order[0]]), 3),
                     expected_p=round(float(p[labels.index(exp)]), 3), rank=order.index(labels.index(exp)) + 1, correct=labels[order[0]] == exp))
with open(os.path.join(ROOT, "evidence", "judge_eval.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
acc = sum(r["correct"] for r in rows) / len(rows)
print(f"judge top-1 accuracy: {sum(r['correct'] for r in rows)}/{len(rows)} = {acc:.1%}")
for r in rows:
    if not r["correct"]:
        print("  miss:", r["run"], r["expected"], "->", r["top1"], r["top1_p"])
