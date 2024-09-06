import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# Flask API endpoint
prediction_endpoint = "http://127.0.0.1:5000/predict"

st.title("Fine food review Analyzer")

uploaded_file = st.file_uploader(
    "Choose a CSV file for bulk prediction - Upload the file and click on Predict",
    type="csv",
)

# Text input for sentiment prediction
user_input = st.text_input("Enter text and click on Predict", "")

# Prediction on single sentence
if st.button("Predict"):
    if uploaded_file is not None:
        try:
            with st.spinner('Predicting...'):
                file = {"file": uploaded_file}
                print("Sending file to prediction endpoint...")
                response = requests.post(prediction_endpoint, files=file)
                response.raise_for_status()
                print("Response received from prediction endpoint.")

                response_bytes = BytesIO(response.content)
                response_df = pd.read_csv(response_bytes)

                st.success('Prediction complete!')
                st.download_button(
                    label="Download Predictions",
                    data=response_bytes,
                    file_name="Predictions.csv",
                    key="result_download_button",
                )
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
            print(f"Error: {e}")
    else:
        try:
            with st.spinner('Predicting...'):
                print(f"Sending text to prediction endpoint: {user_input}")
                response = requests.post(prediction_endpoint, json={"text": user_input})
                response.raise_for_status()
                print("Response received from prediction endpoint.")

                response_json = response.json()
                print(f"Response JSON: {response_json}")

                if 'prediction' in response_json:
                    st.success('Prediction complete!')
                    st.write(f"Predicted sentiment: {response_json['prediction']}")
                else:
                    st.error("Error: 'prediction' key not found in the response")
                    print("Error: 'prediction' key not found in the response")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
            print(f"Error: {e}")

