import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import datetime

def download_data(ticker="NFLX", start_date="2015-01-01"):
    """Download stock data using yfinance."""
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    """Preprocess the stock data."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_dataset(dataset, time_step=1, target_index=3):
    """Create sequences for LSTM input."""
    x_data, y_data = [], []
    for i in range(len(dataset) - time_step - 1):
        x_data.append(dataset[i:i + time_step, :])
        y_data.append(dataset[i + time_step, target_index])
    return np.array(x_data), np.array(y_data)

def build_model(input_shape):
    """Build and compile the LSTM model."""
    model = Sequential([
        LSTM(50, input_shape=input_shape, return_sequences=True),
        LSTM(50, return_sequences=True),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_stock(ticker="NFLX", time_step=100):
    """Complete workflow: download data, train, and predict."""
    # Download and preprocess data
    data = download_data(ticker)
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    scaled_data, scaler = preprocess_data(features)

    # Create train-test split
    training_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:training_size], scaled_data[training_size:]
    x_train, y_train = create_dataset(train_data, time_step)
    x_test, y_test = create_dataset(test_data, time_step)

    # Train the model
    model = build_model(input_shape=(x_train.shape[1], x_train.shape[2]))
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Predict
    test_predict = model.predict(x_test)

    # Inverse scale predictions
    test_predict = scaler.inverse_transform(
        np.concatenate((np.zeros((test_predict.shape[0], scaled_data.shape[1] - 1)), test_predict), axis=1)
    )[:, -1]

    return test_predict, data.index[-len(test_predict):]

