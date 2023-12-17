import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO



def predict_price(stock_name, ahead, days):
    hist = yf.download(stock_name, period="5y")
    # stock = yf.Ticker(stock_name)

    # hist = stock.history(period="5y")

    df=hist
    df = df[['Close']]


    """Scale and reshape the data"""

    scaler = MinMaxScaler()
    df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    # Function to create input sequences and target values for prediction ahead days
    def create_sequences(data, seq_length, ahead=10):
        X, y = [], []
        for i in range(len(data) - seq_length - ahead + 1):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length + ahead - 1])  # Predict 'ahead' days into the future
        return np.array(X), np.array(y)

    # Create sequences and target values
    X, y = create_sequences(df['Close'].values, days, ahead)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    seq_length=days

    model = Sequential()
    #Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 100, return_sequences = True, input_shape = (seq_length, 1)))
    model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 100, return_sequences = True))
    model.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units = 1))
    # Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # Fitting the RNN to the Training set
    model.fit(X_train, y_train, epochs = 10, batch_size = 32)


    # Make predictions
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot actual vs predicted prices
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label='Actual Prices', color='blue')
    plt.plot(predicted_prices, label='Predicted Prices', color='red')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Prices')
    plt.legend()
    plt.xticks(rotation=90)
    STOCK = BytesIO()
    plt.savefig(STOCK, format="png")
    # plt.show()
    STOCK.seek(0)
    plot_img = base64.b64encode(STOCK.getvalue()).decode('utf8')

    # Get the most recent data to predict future prices
    latest_data = df['Close'][-days:].values.reshape(1, -1, 1)

    # Predict the price 'ahead' days in the future
    predicted_price = model.predict(latest_data)
    predicted_price = scaler.inverse_transform(predicted_price)[0][0]

    # Print the predicted price
    print(f"Predicted price {ahead} days from today: {predicted_price:.2f}")

    return predicted_price, plot_img