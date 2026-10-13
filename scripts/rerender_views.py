"""Re-render views/ (8 x 45 deg) and progress/final.png of a finished run from its Gaussian PLY with a given front azimuth.
usage: CUDA_VISIBLE_DEVICES=4 python scripts/rerender_views.py RUN_DIR [--front 90]"""
import argparse, json, math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "dreamgaussian"))
import imageio, numpy as np, torch
from cam_utils import orbit_camera
from gs_renderer import Renderer, MiniCam

p = argparse.ArgumentParser(); p.add_argument("run"); p.add_argument("--front", type=float, default=90); a = p.parse_args()
s = json.load(open(os.path.join(a.run, "summary.json")))
r = Renderer(sh_degree=0); r.initialize(os.path.join(a.run, s["ply"]))
F = math.radians(49.1)

def view(ver, hor, res):
    cam = MiniCam(orbit_camera(ver, hor, 2.5), res, res, F, F, 0.01, 100)
    with torch.no_grad():
        img = r.render(cam, bg_color=torch.ones(3, device="cuda"))["image"]
    return (img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

for h in range(0, 360, 45):
    imageio.imwrite(os.path.join(a.run, "views", f"view_{h:03d}.png"), view(-15, a.front + h, 512))
imageio.imwrite(os.path.join(a.run, "progress", "final.png"), np.concatenate([view(-15, a.front + h, 256) for h in (0, 90, 180, 270)], 1))
s["front_azimuth"] = a.front
json.dump(s, open(os.path.join(a.run, "summary.json"), "w"), indent=2)
print("re-rendered", a.run)
