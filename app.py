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

st.markdown("""
    <style>
        body {
            font-family: "Roboto Mono", monospace;
        }
    </style>
""", unsafe_allow_html=True)

import streamlit as st

# Set the title
st.title("Intelligent Systems - Final Project")

# Introduction
st.write("""
## About This Project
This work was on Streamlit Community Cloud and showcases my work on Machine Learning and Neural Networks.
There are the two Model that's worked on:
- **Titanic Survival Prediction** (Machine Learning) - Kaggle dataset
- **Animal-10 Classification** (Neural Network) - Kaggle dataset
""")

# Navigation buttons
st.write("### Navigate to Projects:")
col1, col2 = st.columns(2)

with col1:
    if st.button("Machine Learning - Titanic Survival"):
        st.switch_page("pages/1_MachineLearning.py")
    elif st.button("Machine Learning Demo - Titanic Survival Demo"):
        st.switch_page("pages/2_MachineLearningDemo.py")

with col2:
    if st.button("Neural Network - Animal 10"):
        st.switch_page("pages/3_NeuralNetwork.py")
    elif st.button("Neural Network Demo - Animal 10 Demo"):
        st.switch_page("pages/4_NeuralNetworkDemo.py")

# Footer
st.write("---")
st.write("6604062630501 Vorapon Witheethum Sec 3")
