import streamlit as st
import pandas as pd

data_path = "data/Titanic.csv" 
df = pd.read_csv(data_path)

def titanic_description():
    st.set_page_config(
        page_title="Machine Learning Model Details",  # Page title
        layout="wide" 
    )

    # Custom Styling
    st.markdown("""
        <style>
            body {
                font-family: "Raleway", sans-serif;
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
            }
            
            .custom-markdown tag{
                color: #FE4A4B;
                font-weight: bold;
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
    """,
    unsafe_allow_html=True
)
        
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
    
    st.subheader("Feature Selection")
    st.write("""
        The following features were selected for model training:
        
        - Pclass (Ticket Class)
        
    """)
    
    with st.expander("⚙️ **Data Preprocessing**", expanded=True):
        st.write("""
        - **Dropped unnecessary columns**: `PassengerId`, `Name`, `Ticket`, `Cabin`, `Embarked`.
        - **Handled missing values**: Missing `Age` values were replaced with the median.
        - **Encoded categorical variables**: Converted `Sex` to numerical values (`0` for female, `1` for male).
        - **Feature selection**: Used `"Pclass", "Sex", "SibSp", "Parch", "Age"` for model training.
        - **Data splitting**: 80% for training, 20% for testing.
        """)
    st.divider()

    # Machine Learning Models
    with st.expander("🏆 **Machine Learning Models Used**", expanded=True):
        st.write("""
        The following models were tested:
        - **Random Forest Classifier** 🌳
        - **K-Nearest Neighbors (KNN)** 🔍
        - **Support Vector Machine (SVM)** 📈

        **Random Forest achieved the highest accuracy and was selected for predictions.**
        """)
    st.divider()

    # Model Evaluation
    with st.expander("📉 **Model Evaluation**", expanded=True):
        st.write("""
        - **Accuracy Score** ✅
        - **Precision, Recall, F1 Score** 📊
        - **Confusion Matrix Visualization** 🟦
        """)

    st.success("Try predicting Titanic passenger survival using the form on this page!")

# Run independently
if __name__ == "__main__":
    titanic_description()
