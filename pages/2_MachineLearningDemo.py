import streamlit as st
import joblib
import torch
import os

# Set page config
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# Load models
model_path = os.path.join("models", "Titanic_random_forest.pkl")
knn_path = os.path.join("models", "Titanic_knn.pkl")
svm_path = os.path.join("models", "Titanic_svm.pkl")

titanic_models = {
    "Random Forest": joblib.load(model_path),
    "K-Nearest Neighbors (KNN)": joblib.load(knn_path),
    "Support Vector Machine (SVM)": joblib.load(svm_path),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# App Title & Description
st.title("Titanic Survival Predictor")
st.markdown("Predict whether a Titanic passenger would survive based on their details.")
st.divider()

# User Input
st.subheader("Passenger Information")
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3])
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
    age = st.number_input("Age", 1, 100, 25)

with col2:
    name = st.text_input("Name", "John Doe")
    sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
    sex = 1 if sex == "Male" else 0
    parch = st.number_input("Parents/Children Aboard (Parch)", 0, 10, 0)

st.divider()

# Model Selection
st.subheader("Model Selection")
selected_model = st.selectbox("Choose a Model", list(titanic_models.keys()))

# Prediction
if st.button("Predict Survival", use_container_width=True):
    input_data = [[pclass, sex, sibsp, parch, age]]
    prediction = titanic_models[selected_model].predict(input_data)
    result = "✅ Survived" if prediction[0] == 1 else "❌ Did NOT Survive"
    
    if(prediction[0] == 1):
        st.success(f"**Prediction:** {name} {result}")
    else:
        st.error(f"**Prediction:** {name} {result}")
        
