import os
import matplotlib.pyplot as plt
import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space

# Sample JSON-style data (you should load this from a file or variable)
dicom_dict = {
    "/home/danieljiang/physionet.org/files/mimic-iv-echo/0.1/files/p10/p10002443/s93330659/93330659_0016.dcm": ["Parasternal_Short"],
    "/home/danieljiang/physionet.org/files/mimic-iv-echo/0.1/files/p10/p10002443/s93330659/93330659_0070.dcm": ["Parasternal_Short"],
    "/home/danieljiang/physionet.org/files/mimic-iv-echo/0.1/files/p10/p10002443/s93330659/93330659_0005.dcm": ["A4C"],
    "/home/danieljiang/physionet.org/files/mimic-iv-echo/0.1/files/p10/p10002443/s93330659/93330659_0025.dcm": ["A2C"],
    "/home/danieljiang/physionet.org/files/mimic-iv-echo/0.1/files/p10/p10002443/s93330659/93330659_0041.dcm": ["Apical_Doppler"],
    # ... add more items here ...
}

# Set up subplot grid
n = min(50, len(dicom_dict))
cols = 10
rows = n // cols + int(n % cols > 0)

fig, axes = plt.subplots(rows, cols, figsize=(20, 2.5 * rows))
axes = axes.flatten()

for idx, (file_path, labels) in enumerate(list(dicom_dict.items())[:50]):
    try:
        dicom_data = pydicom.dcmread(file_path)
        img = dicom_data.pixel_array
        if dicom_data.PhotometricInterpretation == "YBR_FULL_422":
            img = convert_color_space(img, "YBR_FULL_422", "RGB", per_frame=True)[0]
        elif dicom_data.PhotometricInterpretation == "YBR_FULL":
            img = convert_color_space(img, "YBR_FULL", "RGB")[0]
        elif dicom_data.PhotometricInterpretation == "MONOCHROME2":
            pass  # grayscale, no conversion needed
        else:
            print(f"Unsupported photometric interpretation: {dicom_data.PhotometricInterpretation}")
            continue

        axes[idx].imshow(img, cmap='gray' if img.ndim == 2 else None)
        axes[idx].set_title(labels[0])
        axes[idx].axis('off')
    except Exception as e:
        axes[idx].axis('off')
        axes[idx].set_title("Error")
        print(f"Error loading {file_path}: {e}")

# Hide any remaining empty axes
for j in range(idx + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()