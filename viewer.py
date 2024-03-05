import os
import pydicom
import imageio

# Path to the folder containing the patient folders
patients_folder = 'C:/Users/Keola/Documents/GitHub/cardiac-mr-classification/Patients'

# Create a separate folder for patient GIFs
output_folder = os.path.join(patients_folder, 'patient_gifs')
os.makedirs(output_folder, exist_ok=True)

patient_folder = os.path.join(patients_folder, 'SCD0000101')


# Loop through each patient folder
for patient_folder in os.listdir(patients_folder):
    patient_folder_path = os.path.join(patients_folder, patient_folder, "Localizers_1")
    
    # Check if the Localizers_1 folder exists
    if not os.path.isdir(patient_folder_path):
        continue
    
    # Create a list to store the DICOM image paths
    dicom_paths = []
    # Loop through each file in the patient folder
    for file_name in os.listdir(patient_folder_path):
        dicom_paths.append(patient_folder_path + "/" + file_name)
    print(f'Patient #{patient_folder}, {len(dicom_paths)}')
    
    # Sort the DICOM image paths in ascending order
    dicom_paths.sort()
    
    # Create a list to store the image frames
    frames = []
    
    # Loop through each DICOM image path
    for dicom_path in dicom_paths:
        # Read the DICOM image
        dicom_image = pydicom.dcmread(dicom_path)
        
        # Extract the pixel data
        pixel_data = dicom_image.pixel_array
        
        # Append the pixel data to the frames list
        frames.append(pixel_data)
    
    # Create the output GIF or video file path
    output_file = os.path.join(output_folder, f'{patient_folder}.gif')  # Change the extension to .mp4 for video
    
    # Save the frames as a GIF or video
    if frames:
        imageio.mimsave(output_file, frames, duration=0.1)  # Adjust the duration as needed