import os
import json
import glob
import traceback

import torch
import torchvision
import numpy as np
import pydicom
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence
from tqdm import tqdm

import utils
import video_utils

# ─── 1️⃣ CONFIGURATION ──────────────────────────────────────────────────────────
MOUNT_ROOT = os.path.expanduser("~/mount-folder/MIMIC-Echo-IV")
OUTPUT_ROOT = os.path.expanduser("~/inference_output")  # top‐level mirror
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ─── 2️⃣ MODEL & PREPROCESS PARAMS ───────────────────────────────────────────────
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frames_to_take = 32
frame_stride   = 2
video_size     = 224

mean = torch.tensor([29.110628, 28.076836, 29.096405], device=device).reshape(3,1,1,1)
std  = torch.tensor([47.989223, 46.456997, 47.20083],  device=device).reshape(3,1,1,1)


def _serialize_value(val):
    # Bytes → UTF-8 string
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8", errors="ignore")
    # NumPy arrays → lists
    if isinstance(val, np.ndarray):
        return val.tolist()
    # DICOM multi-value → list of serial values
    if isinstance(val, MultiValue):
        return [_serialize_value(v) for v in val]
    # DICOM Sequence → list of dicts
    if isinstance(val, Sequence):
        return [ {**_serialize_dataset(item)} for item in val ]
    # Try to JSON-dump it as is
    try:
        json.dumps(val)
        return val
    except Exception:
        return str(val)

def _serialize_dataset(ds):
    out = {}
    for elem in ds:
        if elem.tag == ds.PixelData.tag:
            continue
        name = elem.name
        out[name] = _serialize_value(elem.value)
    return out

def process_single_dicom(dcm_path):
    # 1) read & extract clean metadata
    ds   = pydicom.dcmread(dcm_path)
    meta = _serialize_dataset(ds)

    # 2) get frames
    pixels = ds.pixel_array
    if pixels.ndim < 3 or (pixels.ndim == 3 and pixels.shape[2] == 3):
        raise ValueError(f"Unexpected pixel dims {pixels.shape}")

    # 3) mask + crop/scale
    pixels = video_utils.mask_outside_ultrasound(pixels)
    x = np.zeros((len(pixels), video_size, video_size, 3), dtype=float)
    for i in range(len(pixels)):
        x[i] = video_utils.crop_and_scale(pixels[i])

    # 4) to torch [C, T, H, W], normalize
    x = torch.as_tensor(x, dtype=torch.float, device=device).permute(3,0,1,2)
    x.sub_(mean).div_(std)

    # 5) pad & stride
    if x.shape[1] < frames_to_take:
        pad = torch.zeros((3, frames_to_take-x.shape[1], video_size, video_size),
                          device=device)
        x = torch.cat([x, pad], dim=1)
    vid = x[:, :frames_to_take:frame_stride, :, :]

    return meta, vid

# ─── 5️⃣ CLASSIFY ONE‐VIDEO TENSOR → VIEW ────────────────────────────────────────
def classify_view(video_tensor):
    # input shape [1, C, T, H, W]
    frames = video_tensor[:, :, 0, :, :].to(device)  # first‐frame batch
    with torch.no_grad():
        logits = view_classifier(frames)
    idx = int(logits.argmax(dim=1).cpu())
    return utils.COARSE_VIEWS[idx]


if __name__ == "__main__":
    # ─── 3️⃣ LOAD VIEW CLASSIFIER ────────────────────────────────────────────────────
    ckpt = torch.load("model_data/weights/view_classifier.ckpt", map_location=device)
    state_dict = {k[6:]: v for k,v in ckpt["state_dict"].items()}

    view_classifier = torchvision.models.convnext_base()
    view_classifier.classifier[-1] = torch.nn.Linear(
        view_classifier.classifier[-1].in_features,
        len(utils.COARSE_VIEWS),
    )
    view_classifier.load_state_dict(state_dict)
    view_classifier.to(device).eval()
    for p in view_classifier.parameters():
        p.requires_grad = False
        
    # ─── 6️⃣ WALK + PROCESS + WRITE JSON ────────────────────────────────────────────
    for root, dirs, files in os.walk(MOUNT_ROOT):
        # collect .dcm files in this folder
        dcms = [f for f in files if f.lower().endswith(".dcm")]
        if not dcms:
            continue

        # build an output folder that mirrors this one
        rel = os.path.relpath(root, MOUNT_ROOT)
        out_folder = os.path.join(OUTPUT_ROOT, rel)
        os.makedirs(out_folder, exist_ok=True)

        results = {}
        for fname in tqdm(dcms, desc=f"Processing {rel}", unit="file"):
            path = os.path.join(root, fname)
            try:
                meta, vid = process_single_dicom(path)
                view = classify_view(vid.unsqueeze(0))
                results[fname] = {
                    "filename": fname,
                    "metadata": meta,
                    "predicted_view": view
                }
            except Exception as e:
                results[fname] = {
                    "error": str(e),
                    "trace": traceback.format_exc()
                }

        # write this folder’s JSON
        out_file = os.path.join(out_folder, "results.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
            print(f"Saved {len(results)} results to {out_file}")

    print("✅ Done — outputs mirrored under", OUTPUT_ROOT)
