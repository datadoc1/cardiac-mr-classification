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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
import random
import torch.nn.functional as F # softmax
from tqdm import tqdm # progress visualisation

import torch
import torchvision
import torchvision.transforms as T # data augmentation
import torchvision.models as models # to get pretrained models
import torch.nn as nn # to build NN, criterion
import torch.optim as optim # optimizer

# plotting and evaluation
from sklearn.metrics import confusion_matrix # performance evaluation

import pandas as pd # read csv
from imblearn.over_sampling import RandomOverSampler as ROS # training data oversampling
from sklearn.model_selection import train_test_split # splitting dataframes
from torch.utils.data import Dataset, DataLoader # data pipeline

from pydicom import dcmread
from PIL import Image

# setting gpu
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# For reproducibility
RANDOM_SEED = 42

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

set_seed(RANDOM_SEED)

# Read HAM10000_metadata.csv
metadata = pd.read_csv('scd_patientdata.csv')

image_dir = 'Patients'
image_files = []
# Read all images in HAM10000_images directory
for patient_folder in os.listdir(image_dir):
    patient_folder_path = os.path.join(image_dir, patient_folder, "Localizers_1")
    
    # Check if the Localizers_1 folder exists
    if not os.path.isdir(patient_folder_path):
        continue

    # Loop through each file in the patient folder
    for file_name in os.listdir(patient_folder_path):
        image_files.append(patient_folder_path + "/" + file_name)

# Label each image with its "dx" from HAM10000_metadata.csv
image_labels = []
for image_file in image_files:
    image_id = image_file.split('\\')[1]
    dx = metadata.loc[metadata['PatientID'] == image_id, 'Pathology'].values[0]
    image_labels.append(dx)

# Create a dictionary to map unique labels to numbers
label_to_number = {label: number for number, label in enumerate(set(image_labels))}

# Convert image_labels to numbers using the dictionary
image_labels = [label_to_number[label] for label in image_labels]

label_counts = {}
for label in image_labels:
    if label in label_counts:
        label_counts[label] += 1
    else:
        label_counts[label] = 1

# Print the number of images associated with each number label
for label, count in label_counts.items():
    print(f"Number of images with label {label}: {count}")

# Print the number of images and their corresponding labels
print(f"Number of images: {len(image_files)}")
print(f"Number of labels: {len(image_labels)}")

# Load the DeiT model
import timm
import time

# Load the models
models_list = [
    ('ResNet', models.resnet18(pretrained=True)),
    ('ViT', timm.create_model('vit_base_patch16_224', pretrained=True)),
    ('DEiT', timm.create_model('deit_small_patch16_224', pretrained=True))
]

# Define the loss function and the optimizer
class_weights = [1 / label_counts[label] for label in range(len(label_counts))]
class_weights = torch.FloatTensor(class_weights).to(DEVICE)
print("Label numbers associated with each class weight:")
for label, weight in zip(range(len(label_counts)), class_weights):
    print(f"Label {label}: {weight.item()}")
criterion = nn.CrossEntropyLoss(weight=class_weights)


# Define the data augmentation
transform = T.Compose([
    T.Resize((224, 224)), # Resize the image to 224x224
    T.Grayscale(num_output_channels=3),  # Convert the image to RGB
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))  # Use single-channel normalization
])

# Define the dataset
class SunnyBrookDataset(Dataset):
    def __init__(self, image_ids, image_labels, transform=None):
        self.image_ids = image_ids
        self.image_labels = image_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        dicom = dcmread(image_id)
        image = Image.fromarray(dicom.pixel_array)  # Convert DICOM data to a PIL Image
        image = image.convert("F")  # Convert the image to float
        label = self.image_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Split the data into training and validation sets
train_image_ids, val_image_ids, train_image_labels, val_image_labels = train_test_split(image_files, image_labels, test_size=0.2, stratify=image_labels, random_state=RANDOM_SEED)

# Create the training and validation datasets
train_dataset = SunnyBrookDataset(train_image_ids, train_image_labels, transform=transform)
val_dataset = SunnyBrookDataset(val_image_ids, val_image_labels, transform=transform)

# Create the training and validation dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

for model_name, model in models_list:
    if model_name == 'ResNet':
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=4, bias=True)
    else:  # ViT and DEiT
        model.head = nn.Linear(in_features=model.head.in_features, out_features=4, bias=True)


# Train and evaluate the models
for model_name, model in models_list:
    NUM_EPOCHS = 10
    
    print(f"Training {model_name} model...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = model.to(DEVICE)

    # Define the optimizer here
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_dataloader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, loss: {running_loss/len(train_dataloader)}")
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training {model_name} model completed in {training_time} seconds.")
    
    # Evaluation metrics
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"Validation accuracy for {model_name} model: {accuracy}")
