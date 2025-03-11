import streamlit as st
from PIL import Image
import torch
from time import sleep
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from stqdm import stqdm
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
    data_dir = "data/raw-img"
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
        
    st.subheader("Dataloader")
    st.write("""
        The DataLoader class is used to create an iterable over the dataset, enabling batch processing and shuffling of the data during training.
    """)
    st.code(""" 
        data_dir = "data/raw-img"
    dataset = CustomImageDataset(root_dir=data_dir, transform=train_transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    """)
    
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
        
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    data_dir = "data/raw-img"
    dataset = CustomImageDataset(root_dir=data_dir, transform=train_transform)
    
    # Check dataset properties
    st.write("Class-to-Index Mapping:", dataset.class_to_idx)
    st.write("Total Images:", len(dataset))
    
    st.markdown('<div class="h2">Model Used</div>', unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    
    st.write("""
        The model used  is a pre-trained ResNet-18 model, which is a convolutional neural network (CNN) architecture that has been trained on the ImageNet dataset.
    """)
    
    st.write("""
       - ##### Residual Connections (Skip Connections):
            The main idea behind ResNet is the use of `residual blocks`. Instead of learning the underlying mapping directly, 
            ResNet learns the residual mapping.
            A residual block adds the input of the block to its output, effectively allowing the network to learn an identity mapping if needed. 
            This helps mitigate the `vanishing gradient problem`, which is common in very deep networks.        
    """)
    
    st.write("""
        - ##### Architecture of ResNet-18:
            `ResNet-18` is a relatively shallow version of the ResNet family, with 18 layers (including convolutional layers, fully connected layers, etc.).
            - ###### It consists of:
                An initial convolutional layer.\n
                A max-pooling layer.\n
                Four residual blocks (each with two convolutional layers).\n
                A global average pooling layer.\n
                A fully connected layer for classification. \n
    """)
    
    st.write("""
        - ##### Residual Blocks:
            Each residual block contains two `3x3 convolutional` layers with batch normalization and ReLU activation.
            The skip connection bypasses these layers and adds the input directly to the output.
    """)
    
    st.write("""
        - ##### Layers Breakdown:
            `Layer 1`: 7x7 convolution with 64 filters, stride 2, followed by max pooling.\n
            `Layer 2`: Two residual blocks with 64 filters.\n
            `Layer 3`: Two residual blocks with 128 filters.\n
            `Layer 4`: Two residual blocks with 256 filters.\n
            `Layer 5`: Two residual blocks with 512 filters.\n
            `Final Layers`: Global average pooling and a fully connected layer for classification.\n
    """)
    
    st.image("data/restnet18_arch.png", caption="ResNet-18 Architecture", width=800)
    
    st.markdown('<div class="h3">Model Setup</div>', unsafe_allow_html=True)
    st.code("""
        # Load the trained model
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    model.load_state_dict(torch.load("Animal10_Restnet18.pth", map_location=device))
    model = model.to(device)
    model.eval()
    """)
    
    st.markdown('<br><br>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)  
    with col1:
        st.code("""
            model = models.resnet18(pretrained=False)
        """)
        
    with col2:
        st.write("""
           - Purpose: Loads a pre-trained ResNet-18 model from `torchvision.models`.
        """)
        st.write("""
            - Details:

               - `pretrained=True`: Initializes the model with weights pre-trained on the ImageNet dataset.
               - ResNet-18 is a convolutional neural network (CNN) architecture with 18 layers, commonly used for image classification tasks
        """)
    
    st.markdown('<br><br>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)  
    with col1:
        st.write("""
            - Purpose: Retrieves the number of input features for the fully connected (FC) layer (fc) of the ResNet-18 model.
        """)
        st.write("""
            - Details:
            
                - The `fc` layer is the final layer of the model, which outputs the predictions for the classification task.
                - For ResNet-18, `num_features` is typically 512.
        """)
        
    with col2:
        st.code("""
           num_features = model.fc.in_features
        """)
        
    st.markdown('<br><br>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)  
    with col1:
        st.code("""
            model.fc = nn.Linear(num_features, 10)
        """)
        
    with col2:
        st.write("""
           - Purpose: Replaces the final fully connected layer of the pre-trained ResNet-18 model with a new FC layer.
        """)
        st.write("""
            - Details:

               - The new FC layer has `num_features` input features and 10 output features (for 10 classes).
               - This modification adapts the pre-trained model to the new task (e.g., classifying images into 10 categories).
        """)
        
    st.markdown('<br><br>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)  
    with col1:
        st.write("""
            - Purpose: Moves the model to the specified device (e.g., GPU or CPU).
        """)
        st.write("""
            - Details:
            
                - `device` is typically set to "cuda" for `GPU` or "cpu" for `CPU`.
        """)
        
    with col2:
        st.code("""
           model = model.to(device)
        """)
        
    col1, col2, col3 = st.columns([0.1,1,0.1]) 
    with col2:
        st.image("data/devide_cuda.png", caption="torch.cuda.is_available()", width=800)
        
    st.markdown('<div class="h3">Loss Function and Optimizer</div>', unsafe_allow_html=True)
    st.code("""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    """)
    
    st.write("""
        - `CrossEntropyLoss` is commonly used for classification tasks. It computes the difference between the predicted class probabilities and the true labels.
        
        - `Adam` is an adaptive optimization algorithm that combines the benefits of momentum and adaptive learning rates.                                                        
        Only the parameters of the final FC layer (`model.fc.parameters()`) are optimized, as the rest of the model uses pre-trained weights.
        The learning rate (`lr`) is set to `0.001`.
    """)
    
    st.markdown('<div class="h3">Training Function</div>', unsafe_allow_html=True)
    st.code("""
            def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=5):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc = evaluate(model, val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0  # Reset counter if accuracy improves
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), "best_model.pth")
        return model        
    """)
    
    st.write("""
        - ##### Parameters:
            - `model`: The model to be trained.

            - `train_loader`: DataLoader for the training dataset.

            - `val_loader`: DataLoader for the validation dataset.

            - `criterion`: The loss function.

            - `optimizer`: The optimizer.

            - `num_epochs`: The maximum number of training epochs (default: 50).
            
            - `patience`: The number of epochs to wait for improvement in validation accuracy before early stopping (default: 5).
        
    """)
    
    st.write("""
        - ##### Steps:
            1. ###### Initialization:
                - best_acc: Tracks the best validation accuracy.

                - best_model_wts: Stores the weights of the best model.

                - early_stop_counter: Counts the number of epochs without improvement in validation accuracy.
    
            2. ###### Training Loop:

                - Iterates over the specified number of epochs.

                - For each epoch:

                    - Sets the model to training mode (`model.train()`).

                    - Iterates through the training data (`train_loader`):

                        - Moves inputs and labels to the specified device.

                        - Computes the model's predictions (`outputs`).

                        - Computes the loss (`criterion(outputs, labels)`).

                        - Performs backpropagation (`loss.backward()`) and updates the model's parameters (`optimizer.step()`).

                        - Tracks the running loss and accuracy.
                    
                - Computes the training accuracy (`train_acc`).

                - Evaluates the model on the validation dataset (`evaluate(model, val_loader)`).

                - Prints the epoch's loss, training accuracy, and validation accuracy.
                
            3. Early Stopping

                - If the validation accuracy improves, the best model weights are saved, and the early stopping counter is reset.

                - If the validation accuracy does not improve for patience epochs, training stops.
            
            4. Saving Best Model

                - The best model weights are loaded back into the model.

                - The best model is saved to a file (best_model.pth).
        """)
    
    st.markdown('<div class="h3">Evaluate Function</div>', unsafe_allow_html=True)
    st.code("""
    def evaluate(model, val_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return correct / total
    """)  
    st.write("""
        - ##### Steps
            1. Sets the model to evaluation mode (`model.eval()`).

            2. Iterates through the validation data (`val_loader`):

                - Moves inputs and labels to the specified device.

                - Computes the model's predictions (`outputs`).

                - Compares the predicted labels with the true labels to compute accuracy.

            3. Returns the `validation accuracy.`
    """)
    
    st.markdown('<div class="h2">Train and Evaluate</div>', unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    st.code("""
        best_model = train_model(model, train_loader, val_loader, criterion, optimizer)
    """)
    
    col1, col2 = st.columns(2)
    with col1 :
        st.image("data/TrainNEvaluate.png", width=400, caption="Train and Evaluate")
    
    with col2:
        st.write("""
            - ##### Purpose: Trains the model using the `train_model` function and saves the best model.

            - ##### Details:

                - The trained model is returned and can be used for further evaluation or inference.
        """)
        
    st.code("""
        tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
    """)
    st.write("""
        - `tqdm` is used to display a progress bar for the training loop.
        - The description (`desc`) includes the current epoch and total epochs.
    """) 
    
    while True:
        num_epoch = 0
        for num_epoch in range(50) :
            for epoch in stqdm(range(50), desc=f"Epoch {num_epoch+1}/{50}", mininterval=0.1, colour="#FE4A4B"):
                sleep(0.1)
        
        sleep(20)
        
if __name__ == "__main__":
    titanic_description()