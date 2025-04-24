import matplotlib.pyplot as plt
import pydicom
from pydicom.pixel_data_handlers import convert_color_space

file_path = r'/Users/lavonda/Downloads/93330659_0016.dcm'

# read in the DICOM with the pydicom module
dicom_data = pydicom.dcmread(file_path)

# print the DICOM metadata
for element in dicom_data:
    print(element)

# note the value for Photometric Interpretation that was printed, it should show:
# (0028, 0004) Photometric Interpretation          CS: 'YBR_FULL_422'
# we need to convert from YBR_FULL_422 to RGB to display the image properly
images_rgb = convert_color_space(dicom_data.pixel_array, "YBR_FULL_422", "RGB", per_frame=True)
# plot the first frame/image
plt.imshow(images_rgb[0])
plt.show()