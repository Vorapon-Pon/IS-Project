import streamlit as st
import joblib
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import pandas as pd
import os
import joblib
import random

# Neural Network Model
animal_model_path = os.path.join("models", "Animal10_Restnet18.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.markdown("""
    <style>
        body {
            font-family: "Roboto Mono", monospace;
        }
    </style>
""", unsafe_allow_html=True)

# Load Animal Classification Model (ResNet18)
class AnimalClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(AnimalClassifier, self).__init__()
        self.model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

animal_model = AnimalClassifier(num_classes=10)

# Load model weights
if os.path.exists(animal_model_path):
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    model.load_state_dict(torch.load(animal_model_path, map_location=device))
    model = model.to(device)
    model.eval()
else:
    raise FileNotFoundError(f"Model file not found: {animal_model_path}")

# Define Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Animal Labels
animal_labels = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "spider", "squirrel"]

dataset_path = "data/raw-img" 

def get_random_image(dataset_path):
    animal_folders = os.listdir(dataset_path)  
    random_animal = random.choice(animal_folders) 
    animal_folder_path = os.path.join(dataset_path, random_animal)
    animal_images = os.listdir(animal_folder_path) 
    random_image = random.choice(animal_images)  
    return os.path.join(animal_folder_path, random_image), random_animal

st.header("Animal-10 Predictior")
st.write("""
    This model can classify images into **10 distinct animal classes**: 
- 🐕 Dog
- 🐎 Horse
- 🐘 Elephant
- 🦋 Butterfly
- 🐔 Chicken
- 🐈 Cat
- 🐄 Cow
- 🐑 Sheep
- 🕷️ Spider
- 🐿️ Squirrel

""")

# Upload Image
uploaded_file = st.file_uploader("Upload an Animal Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
        
    if image.mode == "RGBA":
        image = image.convert("RGB")
        
    st.image(image, caption="Uploaded Image", width=400)

    # Preprocess Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img_tensor = transform(image).unsqueeze(0)  

    # Load Model
    animal_model_path = os.path.join("models", "Animal10_Restnet18.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    model.load_state_dict(torch.load(animal_model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Make Prediction
    with torch.no_grad():
        output = model(img_tensor.to(device))
        predicted_class = torch.argmax(output, dim=1).item()
        
    # Animal Labels (Modify as per dataset labels)
    predicted_animal = animal_labels[predicted_class]

    st.success(f"Prediction: {animal_labels[predicted_class]}")
    
if st.button("Random Image from Dataset"):
    random_image_path, true_label = get_random_image(dataset_path)
    image = Image.open(random_image_path)
    st.image(image, caption=f"True Label: {true_label}", width=400)

    # Preprocess Image
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Make Prediction
    with torch.no_grad():
        output = model(img_tensor.to(device))
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_animal = animal_labels[predicted_class]

    st.success(f"Prediction: {predicted_animal}")