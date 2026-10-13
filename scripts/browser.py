"""Shared headless-Chrome launcher: WebGL2 via ANGLE on the NVIDIA GPU (not SwiftShader)."""
from playwright.sync_api import sync_playwright

CHROME = "/usr/bin/google-chrome"
GPU_FLAGS = ["--use-angle=gl-egl", "--ignore-gpu-blocklist",
             "--no-sandbox"]  # ANGLE/Vulkan lost the context headless; GL-EGL is stable


def launch(p, headless=True):
    return p.chromium.launch(executable_path=CHROME, headless=headless, args=GPU_FLAGS)


if __name__ == "__main__":
    import sys, time
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8549/"
    out = sys.argv[2] if len(sys.argv) > 2 else "shot.png"
    with sync_playwright() as p:
        b = launch(p)
        pg = b.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=1.2)
        logs = []
        pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        pg.on("pageerror", lambda e: logs.append(f"PAGEERROR: {e}"))
        pg.goto(url)
        time.sleep(float(sys.argv[3]) if len(sys.argv) > 3 else 8)
        pg.screenshot(path=out, full_page=True)
        print("HUD:", pg.inner_text("#hud"))
        print("\n".join(logs))
        b.close()
