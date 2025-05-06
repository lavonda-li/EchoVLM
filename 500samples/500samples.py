import os, json
import matplotlib.pyplot as plt
import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space

# Load JSON containing paths
dicom_file = os.path.join(os.path.dirname(__file__), "combined_views.json")
folder_path = "/home/lavonda/mount-folder/MIMIC-Echo-IV"
with open(dicom_file, "r") as f:
    dicom_dict = json.load(f)

# Output folder
output_dir = os.path.join(os.path.dirname(__file__), "dicom_images_output")
os.makedirs(output_dir, exist_ok=True)

for dicom_name, labels in list(dicom_dict.items())[:500]:
    try:
        dicom_path = os.path.join(folder_path, dicom_name)
        dicom_data = pydicom.dcmread(dicom_path)
        img = dicom_data.pixel_array

        if dicom_data.PhotometricInterpretation == "YBR_FULL_422":
            img = convert_color_space(img, "YBR_FULL_422", "RGB", per_frame=True)[0]
        elif dicom_data.PhotometricInterpretation == "YBR_FULL":
            img = convert_color_space(img, "YBR_FULL", "RGB")[0]
        elif dicom_data.PhotometricInterpretation == "MONOCHROME2":
            pass
        else:
            print(f"Unsupported photometric interpretation: {dicom_data.PhotometricInterpretation}")
            continue

        # Create a new figure for each image
        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
        plt.title(labels[0])
        plt.axis('off')

        # Generate filename like: p12_p12715419_s94585416_94585416_0080.dcm.png
        image_name = dicom_name.replace("/", "_")
        image_name = image_name.replace(".dcm", ".png")
        save_path = os.path.join(output_dir, image_name)

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"❌ Error loading {dicom_name}: {e}")
