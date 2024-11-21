import streamlit as st
import requests

# FastAPI URL
API_URL = "http://127.0.0.1:8000/predict/"

# Streamlit app layout
st.title("Netflix Stock Price Prediction")
st.write("Click the button below to get the predicted stock prices for the next few days:")

# Button to trigger prediction
if st.button("Get Prediction"):
    # Call the FastAPI prediction endpoint
    try:
        response = requests.get(API_URL, params={"ticker": "NFLX"})
        data = response.json()

        # Check if the response contains predictions
        if "predictions" in data:
            st.write(f"Predictions for {data['ticker']} stock:")
            predictions = data['predictions']

            # Display the predictions
            for prediction in predictions:
                st.write(f"Date: {prediction['date']}, Predicted Close: ${prediction['predicted_close']:.2f}")
        else:
            st.error(f"Error: {data.get('error', 'Unknown error')}")
    
    except Exception as e:
        st.error(f"Error calling the API: {str(e)}")
