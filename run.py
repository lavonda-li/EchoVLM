# Standard library imports
import os
import math
import glob
import json
import pickle
import argparse

# Third-party library imports
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import pydicom
import sklearn
import sklearn.metrics

# Hugging Face Hub
from huggingface_hub import PyTorchModelHubMixin

# Local module imports
import utils
import video_utils

frames_to_take = 32
frame_stride = 2
video_size = 224
mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)



def process_dicoms(INPUT):
    """
    Reads DICOM video data from the specified folder and returns a dictionary
    mapping filenames to tensors formatted for input into the EchoPrime model.

    Args:
        INPUT (str): Path to the folder containing DICOM files.

    Returns:
        video_dict (dict): A dictionary where keys are filenames and values are
                           tensors representing the video data.
    """
    dicom_paths = glob.glob(f"{INPUT}/**/*.dcm", recursive=True)
    video_dict = {}
    assert len(dicom_paths) > 0, "No DICOM files found in the specified directory."
    for idx, dicom_path in tqdm(enumerate(dicom_paths), total=len(dicom_paths)):
        try:
            # simple dicom_processing
            dcm = pydicom.dcmread(dicom_path)
            pixels = dcm.pixel_array

            # exclude images like (600,800) or (600,800,3)
            if pixels.ndim < 3 or pixels.shape[2] == 3:
                continue

            # if single channel repeat to 3 channels
            if pixels.ndim == 3:
                pixels = np.repeat(pixels[..., None], 3, axis=3)

            # mask everything outside ultrasound region
            pixels = video_utils.mask_outside_ultrasound(dcm.pixel_array)

            # model specific preprocessing
            x = np.zeros((len(pixels), 224, 224, 3))
            for i in range(len(x)):
                x[i] = video_utils.crop_and_scale(pixels[i])

            x = torch.as_tensor(x, dtype=torch.float).permute([3, 0, 1, 2])
            # normalize
            x.sub_(mean).div_(std)

            ## if not enough frames add padding
            if x.shape[1] < frames_to_take:
                padding = torch.zeros(
                    (
                        3,
                        frames_to_take - x.shape[1],
                        video_size,
                        video_size,
                    ),
                    dtype=torch.float,
                )
                x = torch.cat((x, padding), dim=1)

            start = 0
            video_dict[dicom_path] = x[
                :, start : (start + frames_to_take) : frame_stride, :, :
            ]

        except Exception as e:
            print(f"Corrupt file: {dicom_path}")
            print(str(e))

    return video_dict


if __name__ == "__main__":
    # create a argparse for the input folder
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--input",
        type=str,
        help="input folder",
        default="/home/danieljiang/physionet.org/files/mimic-iv-echo/0.1/files",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    vc_checkpoint = torch.load("model_data/weights/view_classifier.ckpt", map_location=device)
    vc_state_dict = {
        key[6:]: value for key, value in vc_checkpoint["state_dict"].items()
    }
    view_classifier = torchvision.models.convnext_base()
    view_classifier.classifier[-1] = torch.nn.Linear(
        view_classifier.classifier[-1].in_features, 11
    )
    view_classifier.load_state_dict(vc_state_dict)
    view_classifier.to(device)
    view_classifier.eval()
    for param in view_classifier.parameters():
        param.requires_grad = False

    video_dict = process_dicoms(args.input)
    output_dict = {}
    for filename, video_tensor in video_dict.items():
        view_list = get_view_list(video_tensor.unsqueeze(0), visualize=False)
        output_dict[filename] = view_list

    # Save the output to a JSON file
    output_file = "view_list_output.json"
    with open(output_file, "w") as f:
        json.dump(output_dict, f, indent=4)
    print(f"View list saved to {output_file}")