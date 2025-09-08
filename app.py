# app.py — Ultra-fast Inference (Images / ZIP / Video) — Dynamo/Inductor disabled
# -------------------------------------------------------------------------------
# - HRNet-W48 head (compatible with your checkpoint)
# - Speed: single-scale, no TTA, pad to /32, optional downscale on long side
# - Video: optional downscale + frame-skip
# - IMPORTANT: TorchDynamo/Inductor disabled (no cl.exe needed on Windows)
#
# Run:  streamlit run app.py

from __future__ import annotations

# ==== Disable TorchDynamo/Inductor BEFORE importing torch ====
import os as _os
_os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")    # disable TorchDynamo
_os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")  # disable Inductor

import os  # <-- later usage (os.path, os.walk, remove, etc.)
import io, zipfile, tempfile, time, math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
from huggingface_hub import snapshot_download  # <-- NEW: HF download

# Optional GPU speed tweaks (no compilation)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ----------- Extra safety: disable dynamo at call site ----------
try:
    import torch._dynamo as _dynamo  # type: ignore
    _DYNAMO_OK = True
except Exception:
    _DYNAMO_OK = False

def _no_dynamo_fn(fn):
    if _DYNAMO_OK:
        return _dynamo.disable(fn)
    return fn

# --------------------------- UI ---------------------------
st.set_page_config(page_title="Corrosion Segmentation — Ultra-fast", layout="wide")
st.title("Corrosion Segmentation — Ultra-fast Inference")

# ----------------------- Model ---------------------------
class HRNetSegHead(nn.Module):
    def __init__(self, variant: str = "hrnet_w48", num_classes: int = 1,
                 pretrained: bool = False, align_corners: bool = False):
        super().__init__()
        self.backbone = timm.create_model(
            variant, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3)
        )
        chs = self.backbone.feature_info.channels()
        self.align_corners = align_corners
        self.lats = nn.ModuleList([nn.Conv2d(c, 64, 1, bias=False) for c in chs])
        self.fuse = nn.Sequential(
            nn.Conv2d(64 * 4, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
        )
        self.cls = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        feats = self.backbone(x)
        tgt = feats[0].shape[-2:]
        ups = [
            F.interpolate(lat(f), size=tgt, mode="bilinear", align_corners=self.align_corners)
            for f, lat in zip(feats, self.lats)
        ]
        f = torch.cat(ups, dim=1)
        f = self.fuse(f)
        out = self.cls(f)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=self.align_corners)
        return out

def _clean_state_dict(sd: dict) -> dict:
    if isinstance(sd, dict) and any(k in sd for k in ("state_dict", "model")):
        sd = sd.get("state_dict", sd.get("model"))
    if isinstance(sd, dict) and sd and all(isinstance(k, str) for k in sd.keys()):
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k.split("module.", 1)[1]: v for k, v in sd.items()}
    return sd

def _strict_load(model: nn.Module, sd: dict):
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(sd.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    if missing or unexpected:
        raise RuntimeError(f"Incompatible checkpoint. Missing={len(missing)} ; Unexpected={len(unexpected)}")
    model.load_state_dict(sd, strict=True)

@st.cache_resource(show_spinner=False)
def get_model(ckpt_path: str) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HRNetSegHead("hrnet_w48", pretrained=False, align_corners=False).to(device)

    # --- NEW: Hugging Face support ("hf://user/repo/path/to/file.pth")
    if ckpt_path and ckpt_path.startswith("hf://"):
        rest = ckpt_path[5:]                     # strip "hf://"
        parts = rest.split("/")
        if len(parts) < 3:
            st.sidebar.error("HF path must be hf://<user>/<repo>/<file>.pth")
            st.stop()
        repo_id = "/".join(parts[:2])            # user/repo
        rel_file = "/".join(parts[2:])           # path in repo

        # token for private repos: from env or Streamlit secrets
        token = os.getenv("HF_TOKEN")
        try:
            token = token or st.secrets.get("HF_TOKEN")
        except Exception:
            pass

        local_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[rel_file],
            token=token
        )
        ckpt_path = os.path.join(local_dir, rel_file)
    # --- END NEW ---

    if not ckpt_path or not os.path.isfile(ckpt_path):
        st.sidebar.error("Checkpoint not found. Please check the path below.")
        return model.eval()
    try:
        raw = torch.load(ckpt_path, map_location=device)
        sd = _clean_state_dict(raw)
        _strict_load(model, sd)
        st.sidebar.success("✅ Checkpoint loaded (strict match)")
    except Exception as e:
        st.sidebar.error(f"❌ Strict load failed: {e}")
        st.stop()
    model.eval()
    return model

# ------------------- Fast inference ---------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

@dataclass
class InferenceCfg:
    img_max_long_side: int = 640      # downscale images (0 = original)
    thr: float = 0.5
    post_close: int = 0
    post_open: int = 0
    fill_holes: bool = False
    overlay_alpha: float = 0.60
    # video
    video_max_long_side: int = 540    # 540p by default
    video_frame_skip: int = 2         # process every Nth frame

CFG = InferenceCfg()

TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def _resize_hw_for_long_side(orig_hw: Tuple[int,int], max_long: int) -> Optional[Tuple[int,int]]:
    if max_long <= 0: return None
    H, W = orig_hw
    long_side = max(H, W)
    if long_side <= max_long: return None
    scale = max_long / float(long_side)
    return (max(1, int(round(H * scale))), max(1, int(round(W * scale))))

def _pad_to_multiple(x: torch.Tensor, mult: int = 32) -> tuple[torch.Tensor, tuple[int,int,int,int]]:
    _, _, H, W = x.shape
    H2 = int(math.ceil(H / mult) * mult)
    W2 = int(math.ceil(W / mult) * mult)
    pad_b = H2 - H
    pad_r = W2 - W
    if pad_b == 0 and pad_r == 0:
        return x, (0,0,0,0)
    x2 = F.pad(x, (0, pad_r, 0, pad_b), mode="replicate")
    return x2, (0, pad_r, 0, pad_b)

def _unpad(x: torch.Tensor, pads: tuple[int,int,int,int]) -> torch.Tensor:
    _, _, H, W = x.shape
    l, r, t, b = pads
    return x[..., 0:H - b, 0:W - r] if (r or b) else x

@_no_dynamo_fn
def _forward_no_dynamo(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return m(x)

@torch.inference_mode()
def predict_prob_map(model: nn.Module, pil_img: Image.Image) -> np.ndarray:
    device = next(model.parameters()).device

    # 1) downscale for speed (network pass)
    H0, W0 = pil_img.height, pil_img.width
    size_hw = _resize_hw_for_long_side((H0, W0), CFG.img_max_long_side)
    if size_hw is None:
        im_in = pil_img
    else:
        Hs, Ws = size_hw
        im_in = pil_img.resize((Ws, Hs), Image.BILINEAR)

    # 2) tensor & /32 padding
    x = TO_TENSOR(im_in).unsqueeze(0).to(device, non_blocking=True)
    if device.type == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)
    x_pad, pads = _pad_to_multiple(x, 32)

    # 3) single-scale forward — NO dynamo/inductor
    use_amp = (device.type == "cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        logit = _forward_no_dynamo(model, x_pad)

    logit = _unpad(logit, pads)

    # 4) prob -> upsample back to original size if downscaled
    prob = torch.sigmoid(logit).squeeze(0).squeeze(0).float().cpu().numpy()
    if size_hw is not None:
        prob = cv2.resize(prob, (W0, H0), interpolation=cv2.INTER_LINEAR)
    return np.clip(prob, 0, 1)

# --------------------- Post-processing -------------------
def morph(mask_u8, close_k=0, open_k=0):
    out = mask_u8.copy()
    if close_k > 0:
        k = 2 * close_k + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ker)
    if open_k > 0:
        k = 2 * open_k + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, ker)
    return out

def fill_holes(mask_u8):
    h, w = mask_u8.shape
    inv = 255 - mask_u8
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(inv, ff, (0, 0), 0)
    return 255 - inv

def to_u8(x01):
    return (np.clip(x01, 0, 1) * 255).astype(np.uint8)

def overlay_red(img_rgb: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    mask01 = (mask_u8 > 0).astype(np.float32)[..., None]
    red = np.zeros_like(img_rgb, dtype=np.float32); red[..., 0] = 255.0
    return np.clip(img_rgb.astype(np.float32) * (1.0 - alpha * mask01) + red * (alpha * mask01), 0, 255).astype(np.uint8)

# ------------------------- Sidebar ------------------------
with st.sidebar:
    st.header("⚙️ Settings (speed first)")
    ckpt = st.text_input(
        "Checkpoint (.pth)",
        value="hf://Jonathan78520/corrosion-hrnet-w48/Checkpoint.pth"  # <-- default to HF
    )
    # Images
    CFG.img_max_long_side = st.number_input("Images: max long side (px, 0=original)", min_value=0, max_value=4000, value=CFG.img_max_long_side, step=64)
    CFG.thr = st.slider("Binary threshold", 0.05, 0.95, float(CFG.thr), 0.01)
    CFG.post_close = st.select_slider("Morphology — Closing", options=[0,1,2,3,4,5], value=CFG.post_close)
    CFG.post_open  = st.select_slider("Morphology — Opening", options=[0,1,2,3,4,5], value=CFG.post_open)
    CFG.fill_holes = st.checkbox("Fill holes", value=CFG.fill_holes)
    CFG.overlay_alpha = st.slider("Overlay alpha", 0.1, 0.9, float(CFG.overlay_alpha), 0.05)
    # Video
    st.markdown("---")
    CFG.video_max_long_side = st.number_input("Video: max long side (px, 0=original)", min_value=0, max_value=4000, value=CFG.video_max_long_side, step=60)
    CFG.video_frame_skip = st.number_input("Video: process every Nth frame", min_value=1, max_value=10, value=CFG.video_frame_skip, step=1)

model = get_model(ckpt)

# -------------------------- Tabs --------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Single Image", "📦 Batch (ZIP of images)", "🎞️ Video"])

# --------- Single Image ---------
with tab1:
    up = st.file_uploader("Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if up:
        pil = Image.open(up).convert("RGB")
        rgb = np.array(pil)

        with st.spinner("Running inference…"):
            prob = predict_prob_map(model, pil)
        pred_u8 = to_u8(prob >= CFG.thr)
        pred_u8 = morph(pred_u8, CFG.post_close, CFG.post_open)
        if CFG.fill_holes:
            pred_u8 = fill_holes(pred_u8)
        ov = overlay_red(rgb, pred_u8, alpha=CFG.overlay_alpha)

        c1, c2, c3 = st.columns(3)
        c1.image(rgb, caption="[1] Input", use_column_width=True)
        c2.image(pred_u8, caption="[2] Zones (binary)", use_column_width=True, clamp=True)
        c3.image(ov, caption="[3] Overlay (red = predicted area)", use_column_width=True)

        colA, colB = st.columns(2)
        _, png_mask = cv2.imencode(".png", pred_u8)
        colA.download_button("⬇️ Binary mask (PNG)", png_mask.tobytes(), "pred_binary.png", "image/png")
        _, png_ov = cv2.imencode(".png", cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        colB.download_button("⬇️ Overlay (PNG)", png_ov.tobytes(), "pred_overlay.png", "image/png")

# --------- Batch (ZIP) ----------
with tab2:
    zup = st.file_uploader("ZIP of images (JPG/PNG)", type=["zip"])
    if zup:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(zup, "r") as zf:
                zf.extractall(td)
            paths: List[str] = []
            for root, _, files in os.walk(td):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        paths.append(os.path.join(root, f))
            paths.sort()
            if not paths:
                st.error("No images found in the ZIP.")
            else:
                st.info(f"{len(paths)} images detected. Running inference…")
                zbuf = io.BytesIO()
                recs = []
                with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as zout:
                    prog = st.progress(0.0)
                    for i, ip in enumerate(paths, start=1):
                        pil = Image.open(ip).convert("RGB")
                        rgb = np.array(pil)
                        prob = predict_prob_map(model, pil)
                        pred_u8 = to_u8(prob >= CFG.thr)
                        pred_u8 = morph(pred_u8, CFG.post_close, CFG.post_open)
                        if CFG.fill_holes:
                            pred_u8 = fill_holes(pred_u8)

                        H, W = pred_u8.shape
                        area_px = int((pred_u8 > 0).sum())
                        area_pct = 100.0 * area_px / (H * W)
                        recs.append({"name": Path(ip).stem, "pred_area_px": area_px, "pred_area_pct": round(area_pct, 3)})

                        _, png = cv2.imencode(".png", pred_u8)
                        zout.writestr(f"pred_binary/{Path(ip).stem}.png", png.tobytes())
                        ov = overlay_red(rgb, pred_u8, alpha=CFG.overlay_alpha)
                        _, pngov = cv2.imencode(".png", cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
                        zout.writestr(f"overlay/{Path(ip).stem}.png", pngov.tobytes())
                        prog.progress(i / len(paths))

                    df = pd.DataFrame.from_records(recs)
                    zout.writestr("summary.csv", df.to_csv(index=False).encode("utf-8"))

                st.dataframe(pd.DataFrame.from_records(recs).sort_values("pred_area_px", ascending=False),
                             use_container_width=True)
                st.download_button("⬇️ Download results (ZIP)", data=zbuf.getvalue(),
                                   file_name="predictions.zip", mime="application/zip")

# --------- Video --------------
with tab3:
    vup = st.file_uploader("Video (MP4/AVI/MOV/MKV)", type=["mp4", "avi", "mov", "mkv"])
    if vup:
        suffix = Path(vup.name).suffix or ".mp4"
        in_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        in_tmp.write(vup.getbuffer())
        in_tmp.flush(); in_tmp.close()
        in_path = in_tmp.name

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            st.error("Failed to open the video.")
            try: os.remove(in_path)
            except Exception: pass
        else:
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

            if CFG.video_max_long_side and max(orig_w, orig_h) > CFG.video_max_long_side:
                scale = CFG.video_max_long_side / float(max(orig_w, orig_h))
                out_w = int(round(orig_w * scale))
                out_h = int(round(orig_h * scale))
            else:
                out_w, out_h = orig_w, orig_h

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out_path = out_tmp.name; out_tmp.close()
            writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
                out_path = out_tmp.name; out_tmp.close()
                writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

            prog = st.progress(0.0)
            i = 0; last_pred = None; t0 = time.time()
            try:
                while True:
                    ret, frame_bgr = cap.read()
                    if not ret: break
                    if (out_w, out_h) != (orig_w, orig_h):
                        frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                    do_infer = (i % max(1, int(CFG.video_frame_skip)) == 0) or (last_pred is None)
                    if do_infer:
                        pil = Image.fromarray(frame_rgb)
                        prob = predict_prob_map(model, pil)
                        pred_u8 = to_u8(prob >= CFG.thr)
                        pred_u8 = morph(pred_u8, CFG.post_close, CFG.post_open)
                        if CFG.fill_holes:
                            pred_u8 = fill_holes(pred_u8)
                        last_pred = pred_u8
                    else:
                        pred_u8 = last_pred

                    ov_rgb = overlay_red(frame_rgb, pred_u8, alpha=CFG.overlay_alpha)
                    writer.write(cv2.cvtColor(ov_rgb, cv2.COLOR_RGB2BGR))

                    i += 1
                    if total > 0: prog.progress(min(1.0, i / total))
            finally:
                cap.release(); writer.release()
                try: cv2.destroyAllWindows()
                except Exception: pass

            with open(out_path, "rb") as f:
                vid_bytes = f.read()

            st.success(f"Done. Frames: {i} | Out: {out_w}×{out_h} | FPS: {fps:.2f} | Time: {time.time()-t0:.1f}s")
            st.video(vid_bytes)
            st.download_button(
                "⬇️ Download overlay video",
                data=vid_bytes,
                file_name=Path(vup.name).stem + ("_overlay.mp4" if out_path.endswith(".mp4") else "_overlay.avi"),
                mime="video/mp4" if out_path.endswith(".mp4") else "video/x-msvideo",
            )

            for p in (in_path, out_path):
                try: os.remove(p)
                except Exception: pass
