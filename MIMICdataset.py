#!/usr/bin/env python3
import os
import json
import traceback
from pathlib import Path

import torch
import torchvision
import numpy as np
import pydicom
from tqdm import tqdm

import utils
import video_utils

# ─── 1️⃣ CONFIG ────────────────────────────────────────────────────────────────
MOUNT_ROOT  = Path(os.path.expanduser("~/mount-folder/MIMIC-Echo-IV"))
OUTPUT_ROOT = Path(os.path.expanduser("~/inference_output"))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DONE_DIRS_FILE = Path("done_dirs.txt")
DONE_DIRS_FILE.touch(exist_ok=True)

# GPU & batching
DEVICE       = torch.device("cuda")
BATCH_SIZE   = 16
MAX_WORKERS  = min(32, os.cpu_count() or 1)

# video‑preprocess params
FRAMES_TAKE  = 32
FRAME_STRIDE = 2
VIDEO_SIZE   = 224
MEAN = torch.tensor([29.110628, 28.076836, 29.096405], device=DEVICE).reshape(3,1,1,1)
STD  = torch.tensor([47.989223, 46.456997, 47.20083],  device=DEVICE).reshape(3,1,1,1)

# ─── 2️⃣ LOAD VIEW CLASSIFIER ──────────────────────────────────────────────────
ckpt = torch.load("model_data/weights/view_classifier.ckpt", map_location=DEVICE)
state_dict = {k[6:]: v for k, v in ckpt["state_dict"].items()}

view_classifier = torchvision.models.convnext_base()
view_classifier.classifier[-1] = torch.nn.Linear(
    view_classifier.classifier[-1].in_features,
    len(utils.COARSE_VIEWS),
)
view_classifier.load_state_dict(state_dict)
view_classifier.to(DEVICE).eval()
for p in view_classifier.parameters():
    p.requires_grad = False

# ─── 3️⃣ DONE_DIRS LOADING ─────────────────────────────────────────────────────
print("🔔 Loading completed folders list …")
DONE_DIRS = set()
if DONE_DIRS_FILE.exists():
    with DONE_DIRS_FILE.open() as f:
        DONE_DIRS.update(line.strip() for line in f if line.strip())
print(f"🔔 {len(DONE_DIRS):,} folders already marked complete.")

# ─── 4️⃣ UTILITIES ─────────────────────────────────────────────────────────────

def _safe(val):
    """Convert DICOM values to JSON‑serializable Python types."""
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


def process_single_dicom(dcm_path: Path):
    dcm   = pydicom.dcmread(dcm_path)
    meta  = {el.name: _safe(el.repval) for el in dcm}
    pixels = dcm.pixel_array

    if pixels.ndim < 3 or pixels.shape[2] == 3:
        raise ValueError(f"Invalid pixel array shape: {pixels.shape}")

    if pixels.ndim == 3:
        pixels = np.repeat(pixels[..., None], 3, axis=3)

    pixels = video_utils.mask_outside_ultrasound(pixels)

    vid_np = np.zeros((len(pixels), VIDEO_SIZE, VIDEO_SIZE, 3), dtype=pixels.dtype)
    for i in range(len(vid_np)):
        vid_np[i] = video_utils.crop_and_scale(pixels[i])

    vid = torch.from_numpy(vid_np).float().permute(3, 0, 1, 2)
    vid.sub_(MEAN).div_(STD)

    if vid.shape[1] < FRAMES_TAKE:
        pad = torch.zeros(3, FRAMES_TAKE - vid.shape[1], VIDEO_SIZE, VIDEO_SIZE)
        vid = torch.cat((vid, pad), 1)

    start = 0
    vid = vid[:, start : start + FRAMES_TAKE : FRAME_STRIDE]
    return meta, vid


def classify_batch(videos: torch.Tensor):
    first_frames = videos[:, :, 0, :, :].to(DEVICE)
    with torch.no_grad():
        logits = view_classifier(first_frames)
    idxs = logits.argmax(1).cpu().tolist()
    return [utils.COARSE_VIEWS[i] for i in idxs]

# ─── 5️⃣ MAIN PIPELINE ─────────────────────────────────────────────────────────

def main():
    print("🚀 Starting MIMIC-Echo-IV inference pipeline …")
    for root, _dirs, files in os.walk(MOUNT_ROOT):
        dcms = [f for f in files if f.lower().endswith(".dcm")]
        if not dcms:
            continue

        rel = os.path.relpath(root, MOUNT_ROOT)

        # Skip if already completed
        if rel in DONE_DIRS:
            print(f"✔️  {rel} already in DONE_DIRS — skipping.")
            continue

        out_folder = OUTPUT_ROOT / rel
        out_folder.mkdir(parents=True, exist_ok=True)
        out_file   = out_folder / "results.json"
        failed_file = out_folder / "failed.txt"

        if out_file.exists():
            print(f"✔️  {rel} already has results.json — adding to DONE_DIRS and skipping.")
            with DONE_DIRS_FILE.open("a") as f:
                f.write(rel + "\n")
            DONE_DIRS.add(rel)
            continue

        # Begin processing this directory
        failed, results = [], {}
        print(f"▶️  Processing {rel}: {len(dcms)} files …")

        metas, vids, names = [], [], []
        for i, name in enumerate(dcms):
            dcm_path = Path(root) / name
            try:
                meta, vid = process_single_dicom(dcm_path)
                metas.append(meta)
                vids.append(vid)
                names.append(name)
            except Exception as e:
                failed.append(os.path.relpath(dcm_path, MOUNT_ROOT))

            # Flush every BATCH_SIZE or at end
            last_item = i == len(dcms) - 1
            if len(vids) >= BATCH_SIZE or (last_item and vids):
                vids_stack = torch.stack(vids)
                views = classify_batch(vids_stack)
                for nm, md, vw in zip(names, metas, views):
                    file_path = Path(root) / nm
                    key = os.path.relpath(file_path, MOUNT_ROOT)
                    results[key] = {"metadata": md, "predicted_view": vw}

                # Write partial results
                with out_file.open("w") as f:
                    json.dump(results, f, indent=2)
                with failed_file.open("w") as f:
                    for fn in failed:
                        f.write(fn + "\n")

                metas.clear(); vids.clear(); names.clear()

        # Mark directory as done
        with DONE_DIRS_FILE.open("a") as f:
            f.write(rel + "\n")
        DONE_DIRS.add(rel)
        print(f"✅  {rel}: successes={len(results)}, failures={len(failed)}")

    print("🎉 All folders processed; outputs saved to", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
