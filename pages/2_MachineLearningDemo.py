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

# Machine Learning Models
model_path = os.path.join("models", "Titanic_random_forest.pkl")
knn_path = os.path.join("models", "Titanic_knn.pkl")
svm_path = os.path.join("models", "Titanic_svm.pkl")

# Load models
titanic_models = {
    "Random Forest": joblib.load(model_path),
    "K-Nearest Neighbors (KNN)": joblib.load(knn_path),
    "Support Vector Machine (SVM)": joblib.load(svm_path),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
