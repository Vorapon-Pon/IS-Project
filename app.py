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

# Use relative paths

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

# Sidebar Navigation
menu = st.sidebar.radio("Choose an Option", ["Machine Learning", "Machine Learning Demo", "Neural Network", "Neural Network Demo" ])

# Titanic Survival Prediction Page
if menu == "Titanic Survival Prediction":
    st.title("📊 Machine Learning Model Details")
    st.write("This page explains the models used in our application.")
    st.subheader("Titanic Passenger Survival Model")
    st.write("""
    - **Random Forest**: An ensemble learning method using multiple decision trees.
    - **K-Nearest Neighbors (KNN)**: A simple algorithm that classifies based on closest data points.
    - **Support Vector Machine (SVM)**: A model that finds the best boundary between different classes.
    """)
# Machine Learning Demo Page
elif menu == "Machine Learning Demo":
    st.title("🚀 Machine Learning Demo")
    st.header("🛳️ Predict Titanic Passenger Survival")
    
    # User Input
    pclass = st.selectbox("Pclass (Ticket Class)", [1, 2, 3])
    name = st.text_input("Name", "John Doe")
    sex = st.radio("Sex", ["male", "female"])
    sex = 1 if sex == "male" else 0
    sibsp = st.number_input("SibSp (Number of Siblings/Spouses Aboard)", 0, 10, 0)
    parch = st.number_input("Parch (Number of Parents/Children Aboard)", 0, 10, 0)
    age = st.number_input("Age", 1, 100, 25)

    # Model Selection
    selected_model = st.selectbox("Choose a Model", list(titanic_models.keys()))

    # Prediction
    if st.button("Predict Survival"):
        input_data = [[pclass, sex, sibsp, parch, age]]
        prediction = titanic_models[selected_model].predict(input_data)
        result = "Survived ✅" if prediction[0] == 1 else "Did NOT Survive ❌"
        st.success(f"Prediction: {result}")
# Neural Network Demo Page
elif menu == "Neural Network Demo":
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