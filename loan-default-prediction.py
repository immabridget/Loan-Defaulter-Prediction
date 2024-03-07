import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.preprocessing import LabelEncoder


background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://miro.medium.com/v2/resize:fit:1100/format:webp/1*ddtswLMjG04yDQoA3Ezw6w.png");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

# Load the trained model
model = joblib.load("best_model1.pkl")

# Define the app title and layout
st.title("Loan Default Prediction App")

# Define input fields for features
credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=600, step=1)
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
income = st.number_input("Income", min_value=0.0, max_value=300000.0, value=10000.0, step=100.0)
LoanAmount= st.number_input("Loan Amount", min_value=5000.0, max_value=250000.0, value=10000.0, step=100.0)
MonthsEmployed = st.number_input("Months Employed", min_value=0, max_value=360, value=1, step=1)
NumCreditLines=st.number_input("Number of Credit Lines",min_value=0,max_value=4,value=1,step=1)
InterestRate=st.number_input("Interest rate",min_value=0.0,max_value=25.0,value=1.0,step=1.0)
DTIRatio=st.number_input("DTI ratio",min_value=0.0,max_value=0.9,value=0.1,step=0.1)
LoanTerm=st.number_input("Loan Term", min_value=0, max_value=60, value=6, step=6)
# HasMortgage = st.selectbox("Has Mortage", ["No","Yes"])
# HasDependents = st.selectbox("Has Dependants", ["No","Yes"])
HasCoSigner = st.selectbox("Has Co-Signer",  ["No","Yes"])
LoanPurpose = st.selectbox("Loan Purpose",  ["Education", "Home", "Auto","Other"])
Education = st.selectbox("Education", ["Bachelor's", "High School", "Master's"])
EmploymentType = st.selectbox("Employment Type", ["Full-time", "Part-time", "self-employed","unemployed"])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single","Divorced"])

# Create a button for making predictions
if st.button("Predict"):
    # Process input values
    input_data = pd.DataFrame(
        {
            "CreditScore": [credit_score],
            "Age": [age],
            "Income": [income],
            "LoanAmount": [LoanAmount],
            "MonthsEmployed":[MonthsEmployed],
            "NumCreditLines": [NumCreditLines],
            "InterestRate": [InterestRate],
            "DTIRatio": [DTIRatio],
            "LoanTerm": [LoanTerm],
            # "HasMortgage": [HasMortgage],
            # "HasDependents": [HasDependents],
            "HasCoSigner": [HasCoSigner],
            "LoanPurpose": [LoanPurpose],
            "Education": [Education],
            "EmploymentType": [EmploymentType],
            "MaritalStatus": [MaritalStatus]
          
        }
    )

    #picking categorical columns and encoding them
    categorical_columns = ['Education', 'EmploymentType', 'MaritalStatus',  'LoanPurpose', 'HasCoSigner']

    # Initializing LabelEncoder
    label_encoder = LabelEncoder()

    # Encoding categorical columns
    for col in categorical_columns:
        input_data[col] = label_encoder.fit_transform(input_data[col])

    # Scale input data using the same scaler used during training
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    # Make a prediction using the trained model
    prediction = model.predict(input_data_scaled)

    # Display the prediction
    if prediction[0] == 1:
        st.success("The customer is at risk of defaulting.")
    else:
        st.success("The customer is not at risk of defaulting.")