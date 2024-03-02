'''
This project is a part of the Sunnybrook Cardiac Data Challenge. The goal of this project is to develop a classifier that can predict the presence of myocardial infarction (MI) in cardiac cine MRI images. The dataset consists of 45 patients with a total of 1200 images. The images are labeled as either healthy, hypertrophy, heart failure having MI or heart failure not having MI. 
The classifier will be trained on a subset of the data and then tested on the remaining data. The model used for the classifier will be an SVM with a radial basis function (RBF) kernel. The features used for the classifier will be extracted from the images using a pre-trained convolutional neural network (CNN). 
The performance of the classifier will be evaluated using the area under the receiver operating characteristic curve (AUC-ROC) and a confusion matrix.

Each patient MRI data is stored in a separate folder. The folder name is the patient ID. Each patient folder contains a number of subfolders, each corresponding to a different MRI sequence. The subfolder names are the sequence names. Each sequence folder contains a number of DICOM files, each corresponding to a single MRI image. The DICOM file names are the image names. The DICOM files contain the image pixel data as well as metadata about the image.
The metadata file (scd_patientdata.csv) contains the following columns:
PatientID: The patient ID
OriginalID: The original patient ID
Gender: The patient gender
Age: The patient age
Pathology: The patient diagnosis
'''

import pydicom
import matplotlib.pyplot as plt
import pandas as pd
import os

# Load in the Metadata
metadata = pd.read_csv('scd_patientdata.csv')

# Specify the directory path
directory = 'C:/Users/Keola/Desktop/SCD_IMAGES_01'

# Get a list of all patient folders
patient_folders = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
patient = patient_folders[0]

# Specify the path to the folder containing manual contours
manual_contours_path = 'C:/Users/Keola/Downloads/scd_manualcontours/SCD_ManualContours'

# Search for "patient" in metadata['PatientID'] and return the OriginalID of that row
original_id = metadata.loc[metadata['PatientID'].str.contains(patient), 'OriginalID'].values[0]
# Create the path for manual contour
manual_contour = os.path.join(manual_contours_path, original_id)




# Specify the path to the folder containing DICOM images
folder_path = 'C:/Users/Keola/Desktop/SCD_IMAGES_01/SCD0000201/CINESAX_300'
# Get a list of all DICOM files in the folder
dicom_files = [file for file in os.listdir(folder_path) if file.endswith('.dcm')]
# Iterate over each DICOM file
for dicom_file in dicom_files:
    # Get the path to the DICOM file
    dicom_file_path = os.path.join(folder_path, dicom_file)
    
    # Read the DICOM image
    dicom_image = pydicom.dcmread(dicom_file_path)
    
    # Display the DICOM image
    plt.imshow(dicom_image.pixel_array, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()
    