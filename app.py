import streamlit as st
import joblib
import numpy as np

# Load the trained classifier model
model = joblib.load('lung_cancer_model.pkl')  # Replace with your actual model file

# Page configuration
st.set_page_config(
    page_title="Lung Cancer Prediction",
    page_icon="💀",
    layout="wide",
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        font-size: 42px;
        color: #D32F2F;
        font-weight: bold;
        text-align: center;
    }
    .sub-title {
        font-size: 20px;
        color: #555555;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #D32F2F;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='main-title'>Lung Cancer Prediction 💀</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Predict the likelihood of lung cancer based on health information.</p>", unsafe_allow_html=True)

# Input fields for the features (limiting to 4 features)
st.header("Health Information")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["M", "F"])
    age = st.slider('Age', min_value=20, max_value=100, value=50, step=1)
with col2:
    smoking = st.selectbox("Do you smoke?", ["Yes", "No"])
    yellow_fingers = st.selectbox("Yellow fingers?", ["Yes", "No"])

# Convert input features into numerical values for prediction
gender = 1 if gender == "M" else 0  # Male=1, Female=0
age = int(age)  # Age remains as an integer
smoking = 1 if smoking == "Yes" else 0  # Yes=1, No=0
yellow_fingers = 1 if yellow_fingers == "Yes" else 0  # Yes=1, No=0

# Create the input data for prediction (only the 4 relevant features)
X = np.array([[gender, age, smoking, yellow_fingers]])

# Predict button
if st.button('Predict'):
    # Use the model to make a prediction based on the input data (independent columns)
    prediction = model.predict(X)

    # Display the result
    st.markdown("### Prediction Result")
    if prediction[0] == 1:
        st.success("The model predicts a **high likelihood** of lung cancer. Please consult a doctor for further tests.")
    else:
        st.success("The model predicts a **low likelihood** of lung cancer. However, please consult a doctor for a professional diagnosis.")

# Footer
st.markdown("<p style='text-align:center; color:#888888; margin-top:50px;'>💀 Powered by Lung Cancer Detection System</p>", unsafe_allow_html=True)
