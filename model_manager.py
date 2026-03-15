"""
model_manager.py
----------------
Central registry for models downloaded at runtime.

Models managed here
    FairFace ONNX : ResNet-34 - race / gender / age (~85 MB)
                    Source: github.com/yakhyo/fairface-onnx

Models NOT managed here
    HSEmotion : installed via pip install hsemotion-onnx (auto-downloads)
"""

import os
import urllib.request
import ssl

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

REGISTRY = {
    "fairface": {
        "path":        os.path.join(MODELS_DIR, "fairface.onnx"),
        "url":         "https://github.com/yakhyo/fairface-onnx/releases/download/weights/fairface.onnx",
        "min_size_mb": 80,
    },
    "emotion": {
        "path":        os.path.join(MODELS_DIR, "enet_b0_8_best_afew.onnx"),
        "url":         "https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx?raw=true",
        "min_size_mb": 15,
    }
}


def get_path(model_name: str) -> str:
    ensure_downloaded(model_name)
    return REGISTRY[model_name]["path"]


def ensure_downloaded(model_name: str) -> bool:
    entry = REGISTRY[model_name]
    path  = entry["path"]
    url   = entry["url"]
    min_b = entry["min_size_mb"] * 1024 * 1024

    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.exists(path) and os.path.getsize(path) >= min_b:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"[ModelManager] {model_name} found ({size_mb:.1f} MB).")
        return True

    print(f"[ModelManager] Downloading {model_name}...")
    print(f"  URL: {url}")
    tmp = path + ".tmp"
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
            with open(tmp, "wb") as fh:
                fh.write(resp.read())
        os.replace(tmp, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"[ModelManager] {model_name} saved ({size_mb:.1f} MB).")
        return True
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f"[ModelManager] Failed to download {model_name}: {exc}\n"
            f"Download manually from:\n  {url}\n"
            f"Place at:\n  {path}"
        ) from exc


def ensure_all() -> None:
    """Download FairFace ONNX. HSEmotion is handled by hsemotion-onnx package."""
    ensure_downloaded("fairface")