from flask import Flask, render_template, send_file, request, flash, redirect, url_for
import MySQLdb.cursors
from flask_mysqldb import MySQL
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os
import math
import matplotlib.pyplot as plt
import pandas as pd
from array import array
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



app = Flask(__name__)
app.secret_key = "stockp"

print("FLASK_ENV:", os.getenv("FLASK_ENV"))


app.config['MYSQL_HOST'] = os.environ.get("MYSQL_HOST")
# app.config['MYSQL_PORT'] = 3306
app.config['MYSQL_USER'] = os.environ.get("MYSQL_USER")
app.config['MYSQL_PASSWORD'] = os.environ.get("MYSQL_PASS")
app.config['MYSQL_DB'] = os.environ.get("MYSQL_DB")

mysql = MySQL(app)

@app.route('/')
def form():
    return render_template('form.html')

""" Once the data is keyed in by the user and the submit button is pressed,
the user will have to wait for the training of the model depending on the
epoch number. Once trained, the model will show the predicted output of this
time series data."""


@app.route('/data', methods=['POST'])
def predict():

    stock_name = request.form['Name']
    # ep = request.form['Epochs']
    ahead = request.form['Ahead']
    days = 30

    # ep = int(ep)
    ahead = int(ahead)
    days = int(days)
    hist = yf.download(stock_name, period="5y")
    # stock = yf.Ticker(stock_name)


    """Parse historical data of 5yrs from Yahoo Finance"""

    # hist = stock.history(period="5y")


    """Create training and test dataset. Training dataset is
    80% of the total data and the remaining 20% will be predicted"""

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
    # seq_length = 3
    X, y = create_sequences(df['Close'].values, days, ahead)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    """Create the model will neural network having 3 layers of LSTM.
    Add the LSTM layers and some Dropout regularisation"""

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
    model.fit(X_train, y_train, epochs = 50, batch_size = 32)


    """Once the model is created, it can be saved. Proceeding forward,
    we need to find out the starting point based on the user inputs for
    "Ahead" and "Days". The data is reshaped next."""


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
    plot_url = base64.b64encode(STOCK.getvalue()).decode('utf8')

    # Get the most recent data to predict future prices
    latest_data = df['Close'][-days:].values.reshape(1, -1, 1)

    # Predict the price 'ahead' days in the future
    predicted_price = model.predict(latest_data)
    predicted_price = scaler.inverse_transform(predicted_price)[0][0]

    # Print the predicted price
    print(f"Predicted price {ahead} days from today: {predicted_price:.2f}")



    """Send the plot to plot.html"""

    # STOCK.seek(0)
    # plot_url = base64.b64encode(STOCK.getvalue()).decode('utf8')
    return render_template("plot.html", plot_url=plot_url, predicted_price=predicted_price, ahead=ahead)





@app.route('/feedback', methods=['POST'])
def feedback():
    print("helloo")
    review_star = request.form['fbstar']
    review_star = float(review_star)
    review_text = request.form['fbtext']
    print(review_star,type(review_star))
    print(review_text)
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('INSERT INTO stock.feedback VALUES (NULL,%s,%s)', (review_star, review_text))
    mysql.connection.commit()
    cursor.close()
    msg = 'Thanks for your feedback!'
    flash(msg,"success")
    return redirect(url_for('form'))

if __name__ == '__main__':
    app.run(debug=True)