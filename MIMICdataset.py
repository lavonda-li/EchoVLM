import os
import json
import glob
import traceback

import torch
import torchvision
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space
from tqdm import tqdm

import utils
import video_utils

# ─── 1️⃣ CONFIGURATION ──────────────────────────────────────────────────────────
MOUNT_ROOT   = os.path.expanduser("~/mount-folder/MIMIC-Echo-IV")
OUTPUT_DIR   = os.path.expanduser("~/inference_output")
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "combined_results.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 2️⃣ MODEL & PREPROCESS PARAMS ───────────────────────────────────────────────
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frames_to_take = 32
frame_stride   = 2
video_size     = 224

mean = torch.tensor([29.110628, 28.076836, 29.096405], device=device) \
            .reshape(3, 1, 1, 1)
std  = torch.tensor([47.989223, 46.456997, 47.20083], device=device) \
            .reshape(3, 1, 1, 1)

# ─── 3️⃣ LOAD VIEW CLASSIFIER ─────────────────────────────────────────────────
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

# ─── 4️⃣ DICOM → Video Tensor ──────────────────────────────────────────────────
def process_dicoms(folder_path):
    dicom_paths = glob.glob(f"{folder_path}/**/*.dcm", recursive=True)
    assert dicom_paths, f"No DICOMs found in {folder_path}"
    video_dict = {}

    for dcm_path in tqdm(dicom_paths, desc="Preprocessing", unit="file"):
        try:
            dcm    = pydicom.dcmread(dcm_path)
            pixels = dcm.pixel_array

            # skip odd shapes / single frames
            if pixels.ndim < 3 or pixels.shape[2] == 3:
                continue

            # mask & crop/scale each frame
            pixels = video_utils.mask_outside_ultrasound(pixels)
            x = np.zeros((len(pixels), video_size, video_size, 3), dtype=float)
            for i in range(len(pixels)):
                x[i] = video_utils.crop_and_scale(pixels[i])

            # to torch [C, T, H, W], normalize
            x = torch.as_tensor(x, dtype=torch.float, device=device) \
                     .permute(3, 0, 1, 2)
            x.sub_(mean).div_(std)

            # pad if too few frames
            if x.shape[1] < frames_to_take:
                pad = torch.zeros(
                    (3, frames_to_take - x.shape[1], video_size, video_size),
                    dtype=torch.float,
                    device=device,
                )
                x = torch.cat([x, pad], dim=1)

            # stride & crop to fixed length
            video_dict[dcm_path] = x[:, :frames_to_take:frame_stride, :, :]

        except Exception as e:
            print(f"⚠️ Corrupt file {dcm_path}: {e}")
            # skip on error

    return video_dict

# ─── 5️⃣ CLASSIFY VIEWS ─────────────────────────────────────────────────────────
def get_view_list(stack_of_videos, visualize=False):
    # stack_of_videos: [B, C, T, H, W]
    first_frames = stack_of_videos[:, :, 0, :, :].to(device)  # [B, C, H, W]
    with torch.no_grad():
        logits = view_classifier(first_frames)
    preds = torch.argmax(logits, dim=1).cpu().tolist()
    view_list = [utils.COARSE_VIEWS[p] for p in preds]
    return view_list

# ─── 6️⃣ MAIN PIPELINE ─────────────────────────────────────────────────────────
all_results = {}

for patient in sorted(os.listdir(MOUNT_ROOT)):
    src_folder = os.path.join(MOUNT_ROOT, patient)
    if not os.path.isdir(src_folder):
        continue

    print(f"\n▶️ Processing folder: {patient}")
    per_folder_out = {}
    out_file = os.path.join(OUTPUT_DIR, f"{patient}_results.json")

    # preprocess all DICOMs in this folder
    video_dict = process_dicoms(src_folder)

    # run inference on each video tensor
    for dcm_path, video_tensor in tqdm(
        video_dict.items(), desc=f"Inference {patient}", unit="file"
    ):
        try:
            views = get_view_list(video_tensor.unsqueeze(0), visualize=False)
            per_folder_out[dcm_path] = views
        except Exception as e:
            per_folder_out[dcm_path] = {
                "error": str(e),
                "trace": traceback.format_exc(),
            }

        # flush every 100 files
        if len(per_folder_out) % 100 == 0:
            with open(out_file, "w") as f:
                json.dump(per_folder_out, f, indent=2)

    # final write for this folder
    with open(out_file, "w") as f:
        json.dump(per_folder_out, f, indent=2)

    # merge into global
    all_results.update(per_folder_out)

# ─── 7️⃣ WRITE COMBINED JSON ───────────────────────────────────────────────────
with open(FINAL_OUTPUT, "w") as f:
    json.dump(all_results, f, indent=2)

print("\n✅ Done!")
print(" Per-folder JSONs in:", OUTPUT_DIR)
print(" Combined JSON at:", FINAL_OUTPUT)
