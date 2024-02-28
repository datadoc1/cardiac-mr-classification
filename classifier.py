import pydicom
import matplotlib.pyplot as plt

# Path to the DICOM file
file_path = r'C:\Users\Kyra\Desktop\Sunnybrook_Cardiac_MR\SCD_IMAGES_01\SCD0000101\CINELAX_301\IM-0004-0001.dcm'

# Read the DICOM file
dcm = pydicom.dcmread(file_path)

# Display the image
plt.imshow(dcm.pixel_array, cmap=plt.cm.gray)
plt.axis('off')
plt.show()