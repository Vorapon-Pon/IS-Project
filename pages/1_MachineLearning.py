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

model_path = os.path.join("models", "Titanic_random_forest.pkl")
knn_path = os.path.join("models", "Titanic_knn.pkl")
svm_path = os.path.join("models", "Titanic_svm.pkl")

# Load models
titanic_models = {
    "Random Forest": joblib.load(model_path),
    "K-Nearest Neighbors (KNN)": joblib.load(knn_path),
    "Support Vector Machine (SVM)": joblib.load(svm_path),
}

def titanic_description():
    st.set_page_config(
        page_title="ML Model Details",  # Page title
        layout="wide" 
    )

    # Data loading and preprocessing
    data_path = "data/Titanic.csv" 
    df = pd.read_csv(data_path)

    columns_to_drop = ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Embarked', 'Fare']

    X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    X['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    X['Age'].fillna(X['Age'].median(), inplace=True)

    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    st.markdown ('<div class="h1">Titanic Survival Prediction </div>', unsafe_allow_html=True)
    st.divider()  # Horizontal line

    # Description
    st.markdown('<div class="h2">Sinking of the Titanic</div>', unsafe_allow_html=True)
    st.write("""
    The sinking of the Titanic is one of the most infamous shipwrecks in history.

    On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

    resulting in the deaths of more than 1,500 people, making it one of the deadliest peacetime maritime disasters in history.
    """)
    
    col1, col2 = st.columns([0.4, 0.4])
    with col1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image("data/Stöwer_Titanic.jpg", caption="The sinking of the Titanic (Illustration by Willy Stöwer, 1912)", width=400)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown(
    """
    <div class="custom-markdown">
        - <tag>Date</tag> : 14–15 April 1912. <br>  
        - <tag>Time</tag> : 23:40–02:20 (02:38–05:18 GMT).  <br>  
        - <tag>Duration</tag> : 2 hours and 40 minutes.  <br>  
        - <tag>Location</tag> : North Atlantic Ocean, 370 miles (600 km) southeast of Newfoundland. <br>    
        - <tag>Coordinates</tag> : 41°43′32″N 49°56′49″W.  <br>  
        - <tag>Cause</tag> : Collision with an iceberg on 14 April.  <br>  
        - <tag>Participants</tag> : Titanic crew and passengers.  <br>  
        - <tag>Outcome</tag> : Maritime policy changes; SOLAS.  <br>  
        - <tag>Deaths</tag> : 1,490–1,635.  <br>  
    </div>
    """,unsafe_allow_html=True)
        
    st.markdown('<div class="h3">How to classify who has a chance of survival?</div>', unsafe_allow_html=True)
    st.write("""
        While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
        in this project, to build a predictive model that answers the question: 
        
        “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
    """)

    # Dataset Overview
    st.markdown('<div class="h2">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    st.write("""
    The dataset was obtained from Kaggle at the [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data).
    and contains the following columns:
    """)
    
    st.subheader("Data Dictionary")
    data_dict = pd.DataFrame({
    "Variable": ["survival", "pclass", "name", "sex", "Age", "sibsp", "parch", "ticket", "fare", "cabin", "embarked"],
    "Definition": [
        "Survival", "Ticket class", "Passenger name", "Sex", "Age in years", 
        "Number of siblings/spouses aboard", "Number of parents/children aboard", 
        "Ticket number", "Passenger fare", "Cabin number", "Port of Embarkation"
    ],
    "Key": [
        "0 = No, 1 = Yes", "1 = 1st, 2 = 2nd, 3 = 3rd", "", "", "", "", "", "", "", "", 
        "C = Cherbourg, Q = Queenstown, S = Southampton"
    ]
    })

    st.table(data_dict)
    
    st.subheader("Understanding the Labels:")
    with st.expander("**Description**", expanded=True):
        st.write("""
        survival:
            
            0 means the passenger did not survive.
            1 means the passenger survived.
        """)
        st.write("""
            pclass (Passenger Class):
            
                This represents the passenger's socio-economic status, indicated by their ticket class.
                
                1 = 1st class (Upper class)
                2 = 2nd class (Middle class)
                3 = 3rd class (Lower class)
                
                Why it matters: Historically, first-class passengers had a higher chance of survival 
                due to their location on the ship (closer to lifeboats) and their social status.

        """)
        st.write("""
            Name (Passenger name):
            
                The passenger's name.                
                
                While the name itself may not be directly predictive,
                analysis of name prefixes (e.g., "Mrs.", "Mr.") could reveal social status or family relationships.
        """)
        st.write("""
            sex (Gender):
            
                This indicates the passenger's gender.
                
                male
                female
                
                Why it matters: "Women and children first" was the evacuation protocol, 
                so gender is a crucial factor in survival.
        """)
        st.write("""
            age (Age in years):
            
                The passenger's age. Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
                
                0 - 100
                
                Why it matters: Children were prioritized during evacuation. 
                Age can also represent physical frailty, which could impact survival.
        """)
        st.write("""
            sibsp (siblings/spouses aboard the Titanic):
                
                This counts the number of siblings or spouses the passenger had on board.

                Sibling = brother, sister, stepbrother, stepsister
                Spouse = husband, wife (mistresses and fiancés were ignored)
            
                Why it matters: It provides information about family size and connections. 
                People traveling with family might have behaved differently during the disaster (e.g., staying together, helping each other)

        """)
        st.write("""
            parch (parents/children aboard the Titanic):
                
                This counts the number of parents or children the passenger had on board.
                
                Parent = mother, father
                Child = daughter, son, stepdaughter, stepson
                Some children travelled only with a nanny, therefore parch=0 for them.
                
                Why it matters: Similar to sibsp, 
                it provides insight into family structure and potential behavior during the disaster.
        """)
        st.write("""
            ticket (Ticket number):
                
                The passenger's ticket number.
            
                While the ticket number itself may not be directly predictive, 
                analysis of ticket prefixes or shared ticket numbers could reveal group travel information.
        """)
        st.write("""
            fare (Passenger fare):
                
                The amount the passenger paid for their ticket.
            
                Why it matters: It's closely related to pclass. 
                Higher fares generally indicate higher social status and better accommodations.
            
        """)
        st.write("""
            cabin (Cabin number):
                
                The passenger's cabin number.
                
                A5 A6 B28 B30 C23 C25 C27 C123 D33 G73 E10 E101 E121 F...
            
                Why it matters: Cabin location could influence access to lifeboats. 
                It could also be used to infer social status. Many cabins contain missing values in the dataset.

        """)
        st.write("""
            embarked (Port of Embarkation):
                
                The port where the passenger boarded the Titanic:
                
                C = Cherbourg
                Q = Queenstown
                S = Southampton
            
                Why it matters: The port of embarkation might correlate with social class or other factors that influenced survival.
        """)
    
    # EDA
    st.markdown('<div class="h2">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    
    st.subheader("Raw Data")
    st.dataframe(df) 
    
    st.subheader("Data Summary")
    st.write(df.describe(include="all"))
    
    st.subheader("Data Preprocessing")
    st.markdown('<div class="h3">Feature Selection</div>', unsafe_allow_html=True)
    st.markdown(
    """
        <div class="custom-markdown">
            historical knowledge tells us that <tag>Gender</tag>, <tag>Age</tag>, <tag>Pclass</tag>, <tag>Sibsp</tag> and <tag>Parch</tag> were significant factors in survival.  <br>
            Therefore, we will use these features for our model training.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="h3 "> Dropped unnecessary columns </div>', unsafe_allow_html=True)
    st.code("df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], inplace=True")
    st.code("# Define features and target\n X = df.drop(columns=['Survived'])\n y = df['Survived']")
    
    st.markdown('<div class="h3">Handling Missing Values</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([0.1, 0.4])
    with col1:
        st.code("df.isnull().sum()")
        st.dataframe(df.isnull().sum())
    
    with col2:
        st.code("df['Age'].fillna(df['Age'].median(), inplace=True)")
        st.write("""
            The missing `Age` values were replaced with the median age.
            """) 
        st.write("""
            For the `Cabin` and `Embarked` column, will drop it as it contains many missing values.
            """) 
        
    st.markdown('<div class="h3">Encoded categorical variables</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([0.4, 0.3])
    with col1:
        st.code("df['Sex'] = LabelEncoder().fit_transform(df['Sex'])")
    
    with col2:
        st.write("""
            Encoded categorical variables to Numerical Converted Sex (Gender) (0 for female, 1 for male).
        """)
    
    st.markdown('<div class="h3">Data splitting</div>', unsafe_allow_html=True)
    st.code("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)")
    st.write("""
        Split the data into 80% for training and 20% for testing.
    """)    
    st.dataframe(X.head())
    st.divider()

    # Models Used 
    st.markdown('<div class="h2"> Models Used </div>', unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)

    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.subheader("K-Nearest Neighbors (KNN)")
        st.write("&emsp;&emsp;KNN is a simple algorithm that classifies data points based on the majority class of their k-nearest neighbors. It is a non-parametric method that does not make strong assumptions about the underlying data distribution.")
    with col2:
        st.subheader("Hyperparameters")
        st.markdown("""
        - `n_neighbors` : Number of neighbors to consider for classification.
        - `Distance Metric` : The measure used to calculate the distance between data points (e.g., Euclidean, Manhattan).
        """)
    
    st.markdown('<br>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.subheader("Support Vector Machine (SVM)")
        st.write("&emsp;&emsp;SVM is a powerful algorithm that finds the optimal hyperplane to separate data points into different classes. It works well for both linearly separable and non-linearly separable data.")

    with col2:
        st.subheader("Hyperparameters")
        st.markdown("""
        - `C` : Regularization parameter that controls the trade-off between margin maximization and error minimization.
        - `Kernel` : The function used to transform the input data into a higher-dimensional space.
        """)
    
        st.markdown('<br>', unsafe_allow_html=True)

    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.subheader("Random Forest Classifier")
        st.write("&emsp;&emsp;Random Forest is an ensemble learning method that builds multiple decision trees and combines their predictions to improve classification performance. Each tree is trained on a random subset of the data and features, reducing overfitting and increasing robustness.")

    with col2:
        st.subheader("Hyperparameters")
        st.markdown("""
        - `n_estimators` : Number of decision trees in the forest.
        - `max_depth` : Maximum depth of each decision tree.
        """)
        
    st.subheader("Model Details")
    
    st.markdown('<div class="h3">Random Forest</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.code("RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)\nmodel.fit(X_train, y_train)")
        
    with col2:
        st.write("""
            An ensemble learning method using multiple decision trees. 
            in this model used 100 decision trees with a maximum depth 5.
        """)
        
    st.markdown('<div class="h3">K-Nearest Neighbors (KNN)</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.code("knn_model = KNeighborsClassifier(n_neighbors=5)\nknn_model.fit(X_train, y_train)")
    
    with col2:
        st.write("""
            A simple algorithm that classifies based on the majority class of their k-nearest neighbors. 
            in this model used 5 neighbors for classification.
        """)
        
    st.markdown('<div class="h3">Support Vector Machine (SVM)</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.code("svm_model = SVC(C=1.0, kernel='rbf', random_state=42)\nsvm_model.fit(X_train, y_train)")
    
    with col2:
        st.write("""
            A model that finds the best boundary between different classes. 
            in this model used a radial basis function kernel with a regularization parameter of 1.0.
        """)
    
    st.markdown('<div class="h2">Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)

    st.code("""
# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

print('Classification Report:')
print(classification_report(y_test, y_pred))

print('Confusion Matrix:')
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
            """)
   
    col1, col2 = st.columns([0.3, 0.5])
    with col1:
        # Model selection
        model_choice = st.selectbox("Choose a Model", list(titanic_models.keys()))
        model = titanic_models[model_choice]
        
        # Ensure the feature names match those used during fitting
        X_test = X_test[model.feature_names_in_]
        y_pred = model.predict(X_test)

        # Model evaluation
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f'### Accuracy: {accuracy:.4f}')

        st.write("### Classification Report:")
        st.text(classification_report(y_test, y_pred))
   
    with col2:
        st.write("### Confusion Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
     
    st.success("Try predicting Titanic passenger survival using the form on Demo page!")

# Run independently
if __name__ == "__main__":
    titanic_description()
