import streamlit as st
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
from torch import optim
from torchvision import models
from PIL import Image
import os

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
    
    st.code("""
    class CustomImageDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.image_paths = []

            for cls in self.classes:
                class_path = os.path.join(root_dir, cls)
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    self.image_paths.append((img_path, self.class_to_idx[cls]))

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path, label = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")  # Open image & convert to RGB

            if self.transform:
                image = self.transform(image)

            return image, label

    """)
    
    st.write("""
        This class inherits from `torch.utils.data.Dataset`, which is a base class for datasets in PyTorch. It is used to load and preprocess image data.
    """)     
    
    st.markdown('<div class="h3">__init__ Method</div>', unsafe_allow_html=True)
    st.write("""
        - Purpose: Initializes the dataset.\n- Parameters:
        \n\troot_dir: The root directory where the dataset is stored. It is assumed that the directory is organized in subdirectories, where each subdirectory corresponds to a class (e.g., root_dir/class1, root_dir/class2, etc.).
        \n\ttransform: A function or a composition of transformations (e.g., resizing, flipping, normalization) to be applied to the images.
    """)
    st.code("""
     # Initialize dataset
    data_dir = "IS-Project/data/raw-img"
    dataset = CustomImageDataset(root_dir=data_dir, transform=train_transform)
    """)
    st.write("""Steps:

        1.Stores the root_dir and transform.

    2.Lists all subdirectories in root_dir (each representing a class) and sorts them alphabetically.

    3.Creates a mapping (class_to_idx) from class names to numerical indices (e.g., {"class1": 0, "class2": 1}).

    4.Iterates through each class directory, collects the paths of all images, and stores them in image_paths along with their corresponding class index.
    """)
    
    st.markdown('<div class="h3">__len__ Method</div>', unsafe_allow_html=True)
    st.write("""
        - Purpose: Returns the total number of images in the dataset.
    """)
    
    st.markdown('<div class="h3">__getitem__ Method</div>', unsafe_allow_html=True)
    st.write("""
        - Purpose: Loads and returns an image and its corresponding label for a given index.
    """)
    
    st.write("""Steps:

        1.Retrieves the image path and label for the given index (idx).

    2.Opens the image using `PIL.Image.open` and converts it to RGB format.

    3.Applies the specified transformations (if any) to the image.

    4.Returns the transformed image and its label.
    """)
    
    st.subheader("Transformations")
    st.write("""
        The train_transform is a composition of image transformations defined using `torchvision.transforms.Compose`. 
        These transformations are applied to the images during training to augment the dataset and improve model generalization.
    """)
    st.code("""
            # Define transformations
        train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]))
    """, language="python")
    
    st.markdown('<div class="h3">Transformations Applied:</div>', unsafe_allow_html=True)
    st.write("""
            
        1. Resize((224, 224)): Resizes the image to 224x224 pixels.

    2. RandomHorizontalFlip(): Randomly flips the image horizontally with a default probability of 0.5.

    3. RandomRotation(30): Randomly rotates the image by up to 30 degrees.

    4. ColorJitter(...): Randomly changes the brightness, contrast, saturation, and hue of the image.

    5. ToTensor(): Converts the image from a PIL image or NumPy array to a PyTorch tensor and scales pixel values to the range [0, 1].

    6. Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]): Normalizes the tensor by subtracting the mean (0.5) and dividing by the standard deviation (0.5) for each channel (R, G, B). This scales the pixel values to the range [-1, 1].
    """)
    
    image_path = "data/raw-img/gatto/67.jpeg"  # Change to your actual path
    original_image = Image.open(image_path)
    
    col1, col2, col3 = st.columns([0.35,0.2,0.35])
    with col1:
        st.image(original_image, caption="Original Image",width=250)
    
    if "tensor_img" not in st.session_state:
        st.session_state.tensor_img = original_image
           
    with col2:
        st.markdown('<br><br>', unsafe_allow_html=True)
        st.header("&emsp;  ->")
        if st.button("Apply Transformation"):
            # Define transformations
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.ToPILImage()
            ])

            # Apply transformations
            tensor_img = train_transform(original_image)
            
             # Convert tensor back to image
            st.session_state.tensor_img = tensor_img
            
    with col3:
        st.image(st.session_state.tensor_img, width=250, caption="Transformed Image")
        
    
if __name__ == "__main__":
    titanic_description()