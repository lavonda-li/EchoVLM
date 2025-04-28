#!/usr/bin/env python3
import os
import json
import traceback

import torch
import torchvision
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space
from tqdm import tqdm

import utils
import video_utils

# ─── 1️⃣ CONFIG ────────────────────────────────────────────────────────────────
MOUNT_ROOT   = os.path.expanduser("~/mount-folder/MIMIC-Echo-IV")
OUTPUT_ROOT  = os.path.expanduser("~/inference_output")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# GPU & batching
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 16
MAX_WORKERS  = min(32, os.cpu_count() or 1)

# video preprocess params
frames_to_take = 32
frame_stride   = 2
video_size     = 224
mean = torch.tensor([29.110628, 28.076836, 29.096405], device=device).reshape(3,1,1,1)
std  = torch.tensor([47.989223, 46.456997, 47.20083],  device=device).reshape(3,1,1,1)

# ─── 2️⃣ LOAD VIEW CLASSIFIER ──────────────────────────────────────────────────
ckpt = torch.load("model_data/weights/view_classifier.ckpt", map_location=device)
state_dict = {k[6:]: v for k, v in ckpt["state_dict"].items()}

view_classifier = torchvision.models.convnext_base()
view_classifier.classifier[-1] = torch.nn.Linear(
    view_classifier.classifier[-1].in_features,
    len(utils.COARSE_VIEWS),
)
view_classifier.load_state_dict(state_dict)
view_classifier.to(device).eval()
for p in view_classifier.parameters():
    p.requires_grad = False

# ─── 3️⃣ SAFE METADATA SERIALIZATION ───────────────────────────────────────────
def _safe(val):
    """Convert DICOM values to JSON-serializable Python types."""
    import numpy as _np
    if isinstance(val, (bytes, bytearray)):
        return val.decode(errors="ignore")
    if isinstance(val, _np.ndarray):
        return val.tolist()
    try:
        json.dumps(val)
        return val
    except Exception:
        return str(val)

# ─── 4️⃣ DICOM → TENSOR + METADATA ─────────────────────────────────────────────
def process_single_dicom(dcm_path):
    ds = pydicom.dcmread(dcm_path)
    # JSON-safe metadata dict
    meta = {elem.name: _safe(elem.value) for elem in ds if elem.name != "PixelData"}

    pixels = ds.pixel_array
    if pixels.ndim < 3 or (pixels.ndim == 3 and pixels.shape[2] == 3):
        raise ValueError(f"Unexpected pixel dims {pixels.shape}")

    pixels = video_utils.mask_outside_ultrasound(pixels)
    x = np.zeros((len(pixels), video_size, video_size, 3), dtype=float)
    for i in range(len(pixels)):
        x[i] = video_utils.crop_and_scale(pixels[i])

    x = torch.as_tensor(x, dtype=torch.float, device=device).permute(3,0,1,2)
    x.sub_(mean).div_(std)
    if x.shape[1] < frames_to_take:
        pad = torch.zeros((3, frames_to_take - x.shape[1], video_size, video_size),
                          device=device)
        x = torch.cat([x, pad], dim=1)
    video = x[:, :frames_to_take:frame_stride, :, :]
    return meta, video

# ─── 5️⃣ BATCH CLASSIFY ─────────────────────────────────────────────────────────
def classify_batch(videos):
    # videos: [B, C, T, H, W]
    first_frames = videos[:, :, 0, :, :].to(device)
    with torch.no_grad():
        logits = view_classifier(first_frames)
    idxs = logits.argmax(dim=1).cpu().tolist()
    return [utils.COARSE_VIEWS[i] for i in idxs]

# ─── 6️⃣ MAIN PIPELINE ─────────────────────────────────────────────────────────
def main():
    for root, dirs, files in os.walk(MOUNT_ROOT):
        dcms = [f for f in files if f.lower().endswith(".dcm")]
        if not dcms:
            continue

        rel        = os.path.relpath(root, MOUNT_ROOT)
        out_folder = os.path.join(OUTPUT_ROOT, rel)
        os.makedirs(out_folder, exist_ok=True)

        out_file = os.path.join(out_folder, "results.json")
        failed_file = os.path.join(out_folder, "failed.txt")

        if os.path.exists(out_file):
            with open(out_file) as f:
                results = json.load(f)
        else:
            results = {}

        processed = set(results.keys())
        to_do     = [f for f in dcms if f not in processed]

        if not to_do:
            print(f"✔️  {rel} already done—found {out_file}. Skipping.")
            continue

        failed = []
        print(f"\n▶️  Processing {rel}: {len(to_do)} files remaining")

        metas, vids, names = [], [], []
        for i, name in enumerate(to_do):
            dcm_path = os.path.join(root, name)
            try:
                meta, vid = process_single_dicom(dcm_path)
                metas.append(meta)
                vids.append(vid)
                names.append(name)
            except Exception as e:
                results[name] = {"error": str(e), "trace": traceback.format_exc()}
                failed.append(name)

            # Every BATCH_SIZE files, classify + flush
            if len(vids) >= BATCH_SIZE or (i == len(to_do) - 1):
                vids_stack = torch.stack(vids)
                views = classify_batch(vids_stack)
                for nm, md, vw in zip(names, metas, views):
                    results[nm] = {"metadata": md, "predicted_view": vw}

                # Save results
                with open(out_file, "w") as f:
                    json.dump(results, f, indent=2)
                with open(failed_file, "w") as f:
                    for fn in failed:
                        f.write(fn + "\n")

                # Clear buffers for next batch
                metas.clear()
                vids.clear()
                names.clear()

        successes = len(results) - len(failed)
        print(f"✅  {rel}: done. successes={successes}, failures={len(failed)}")

    print("\n🎉 All folders processed; outputs under", OUTPUT_ROOT)