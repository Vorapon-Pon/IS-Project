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

# Neural Network Model
animal_model_path = os.path.join("models", "Animal10_Restnet18.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

st.header("🐶 Classify an Animal Image")

# Upload Image
uploaded_file = st.file_uploader("Upload an Animal Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
        
    if image.mode == "RGBA":
        image = image.convert("RGB")
        
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

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
    animal_labels = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "spider", "squirrel"]
    predicted_animal = animal_labels[predicted_class]

    st.success(f"Prediction: {predicted_class} ({animal_labels[predicted_class]})")