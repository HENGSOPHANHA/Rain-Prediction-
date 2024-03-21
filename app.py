import streamlit as st
import numpy as np
from joblib import load

# Load the logistic regression weights and the scaler
weights = load('logistic_regression_weights.joblib')
scaler = load('scaler.joblib')


# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Define the predict function
def predict(X, weights):
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add intercept
    return sigmoid(np.dot(X, weights)) >= 0.5  # Make a prediction


# Streamlit application
def run_app():
    # Application title
    st.title('Rain Prediction in Cambodia')

    # Input fields for features
    # Initializing an empty list for feature inputs
    feature_input = []

    # Defining input fields for all specified features
    humidity = st.number_input('Humidity:', value=50.0)
    feature_input.append(humidity)

    cloudcover = st.number_input('Cloud Cover:', value=50.0)
    feature_input.append(cloudcover)

    tempmin = st.number_input('Minimum Temperature:', value=25.0)
    feature_input.append(tempmin)

    feelslikemax = st.number_input('Maximum Feels Like Temperature:', value=35.0)
    feature_input.append(feelslikemax)

    feelslikemin = st.number_input('Minimum Feels Like Temperature:', value=25.0)
    feature_input.append(feelslikemin)

    feelslike = st.number_input('Average Feels Like Temperature:', value=30.0)
    feature_input.append(feelslike)

    dew = st.number_input('Dew Point:', value=20.0)
    feature_input.append(dew)

    temp = st.number_input('Average Temperature:', value=28.0)
    feature_input.append(temp)

    precip = st.number_input('Precipitation:', value=0.0)
    feature_input.append(precip)

    windgust = st.number_input('Wind Gust:', value=10.0)
    feature_input.append(windgust)

    windspeed = st.number_input('Wind Speed:', value=5.0)
    feature_input.append(windspeed)

    winddir = st.number_input('Wind Direction:', value=180.0)
    feature_input.append(winddir)

    sealevelpressure = st.number_input('Sea Level Pressure:', value=1010.0)
    feature_input.append(sealevelpressure)

    tempmax = st.number_input('Maximum Temperature:', value=35.0)
    feature_input.append(tempmax)

    visibility = st.number_input('Visibility:', value=10.0)
    feature_input.append(visibility)

    solarradiation = st.number_input('Solar Radiation:', value=200.0)
    feature_input.append(solarradiation)

    solarenergy = st.number_input('Solar Energy:', value=5.0)
    feature_input.append(solarenergy)

    uvindex = st.number_input('UV Index:', value=5)
    feature_input.append(uvindex)

    # Button to make prediction
    if st.button('Predict Rain'):
        # Scale the input features
        scaled_features = scaler.transform([feature_input])
        # Make a prediction
        prediction = predict(scaled_features, weights)

        # Display result
        if prediction:
            st.success('Rain is likely. ☔')
        else:
            st.success('No rain expected. ☀️')

if __name__ == '__main__':
    run_app()
