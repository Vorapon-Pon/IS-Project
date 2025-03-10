import streamlit as st
import pandas as pd
import seaborn as sns
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def titanic_description():
    st.set_page_config(
        page_title="NN Model Details",  # Page title
        layout="wide" 
    )

    # Data loading and preprocessing
    data_path = "data/raw-img" 

    # Custom Styling
    st.markdown("""
        <style>
            body {
                font-family: "Roboto Mono", monospace;
            }
            /* Center content and limit max width */
            .block-container {
                padding-left: 5rem;
                padding-right: 5rem;
                max-width: 80% !important;
            }

            /* Reduce sidebar width for better responsiveness */
            [data-testid="stSidebar"] {
                width: 250px;
            }

            /* Hide sidebar toggle button when sidebar is collapsed */
            @media (max-width: 768px) {
                [data-testid="stSidebar"] {
                    display: none;
                }
            }
            
            .image-caption {
                font-size: 14px;
                font-style: italic;
                color: gray;
            }

            /* Center align headers */
            .h1 {
            text-align: center;
            font-size: 48px;
            font-weight: bold;
            }

            .h2 {
            font-size: 36px;
            padding-top: 30px;
            }

            .h3 {
            font-size: 24px;
            padding-top: 10px;
            }
            
            .custom-markdown tag{
                color: #FE4A4B;
            }

        </style>
    """, unsafe_allow_html=True)

    # Page Title
    st.markdown ('<div class="h1">Animal-10 Prediction </div>', unsafe_allow_html=True)
    st.divider()  # Horizontal line

    # Description
    st.markdown('<div class="h2">Animal classification</div>', unsafe_allow_html=True)
    st.markdown(
    """
        <div class="custom-markdown">
            The selection of these ten classes was deliberate, aiming to provide a balance between common domestic animals 
            (like <tag>dogs</tag>, <tag>cats</tag>, <tag>cows</tag>, and <tag>sheep</tag>), 
            larger mammals (<tag>horses</tag> and <tag>elephants</tag>), insects (<tag>butterflies</tag> and <tag>spiders</tag>), and avian creatures 
            (<tag>chickens</tag>), alongside the agile <tag>squirrel</tag>. This variety ensures that models trained on this dataset must learn to differentiate 
            between subtle and significant visual differences, from the textural variations of fur and feathers to the distinct body shapes and poses characteristic of each species.
        </div>
        <div style='padding-top: 10px; padding-bottom: 10px;'>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("data/dog.jpg", caption="Dog", width=200)
    
    with col2:
        st.image("data/cat.jpg", caption="Cat", width=200)
    
    with col3:
        st.image("data/hamster.gif", caption="Hamter", width=200)
    
    st.markdown('<div class="h2">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    st.write("""
        This is the dataset contains about 28K medium quality animal images belonging to 10 categories: dog, cat, horse, spyder, butterfly, chicken, sheep, cow, squirrel, elephant.

    All the images have been collected from "google images" and have been checked by human. There is some erroneous data to simulate real conditions (eg. images taken by users of your app).

    The main directory is divided into folders, one for each category. Image count for each category varies from 2K to 5 K units.

    """)
    
    st.write("""
        The dataset was obtained from the Kaggle at the [Animal-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10/data) and contains 10 classes of animals.         
    """)
    
    st.subheader("Dataset Structure")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("data/dataset_structure.png", width=400)
    
    with col2: 
        st.write("""
            The dataset contains 10 folders, one for each class of animals. Each folder contains images of the respective class. 
            The dataset is structured as follows:
        """)
        
    st.markdown('<div class="h3">Example Images</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="h3">cane (dog)</div>', unsafe_allow_html=True)
    st.header("&emsp;|  ->")
    
    with st.expander("Show Images", expanded=True):
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image("data/raw-img/cane/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg", caption="Dog 1", width=180)
            st.image("data/raw-img/cane/OIP-_2Itmpob3Q0nbJKrHvtnfAHaJ3.jpeg", caption="Dog 5", width=180)

        with col2:
            st.image("data/raw-img/cane/OIP-__Yu1XH3iAC10OzGQFpC-AHaE8.jpeg", caption="Dog 2", width=180)
            st.image("data/raw-img/cane/OIP-_3acmW_iSr12XgQTNz0IdQHaFj.jpeg", caption="Dog 6", width=180)
            

        with col3:
            st.image("data/raw-img/cane/OIP-_-AtcUnGMN6ht4EYKmgkXgHaE8.jpeg", caption="Dog 3", width=180)
            st.image("data/raw-img/cane/OIP-_3S-iEDMQnko7ZHgq_FTcwHaEL.jpeg", caption="Dog 7", width=180)
            
        with col4:
            st.image("data/raw-img/cane/OIP-_2iBsOsobKZsP76-9Cd-qAHaEM.jpeg", caption="Dog 4", width=180)
            st.image("data/raw-img/cane/OIP-_4M8lLVlk06o0YOtolSlvQHaHL.jpeg", caption="Dog 8", width=180)
    
    st.markdown('<div class="h3">gatto (cat)</div>', unsafe_allow_html=True)
    st.header("&emsp;|  ->")
    
    with st.expander("Show Images", expanded=True):
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image("data/raw-img/gatto/97.jpeg", caption="Cat 1", width=180)
            st.image("data/raw-img/gatto/148.jpeg", caption="Cat 5", width=180)

        with col2:
            st.image("data/raw-img/gatto/12.jpeg", caption="Cat 2", width=180)
            st.image("data/raw-img/gatto/67.jpeg", caption="Cat 6", width=180)
            

        with col3:
            st.image("data/raw-img/gatto/163.jpeg", caption="Cat 3", width=180)
            st.image("data/raw-img/gatto/226.jpeg", caption="Cat 7", width=180)
            
        with col4:
            st.image("data/raw-img/gatto/299.jpeg", caption="Cat 4", width=180)
            st.image("data/raw-img/gatto/328.jpeg", caption="Cat 8", width=180)
            
    st.markdown('<div class="h2">Preprocessing images</div>', unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)      
    
if __name__ == "__main__":
    titanic_description()