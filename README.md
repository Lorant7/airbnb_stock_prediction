# Stock Price Prediction
This project trains and tests a model to predict if the value of a stock will increase or decrease on the following day based only on numerical data.

---

## Introduction
The goal of this project was to prove (mainly to myself) that the stock price variation is too volatile to be predicted with the numerical by simply looking at the stock's history. To prove this, I crated this package that trains either a Recursive Neural Network(RNN) or a Long Short Term Memory(LSTM) deep neural network using features such as opening and closing value of the stock, highest and lowest value each day, and the time of year to make a prediction. 

---

## Features
- Acquires up-to-date data from yahoo finance.
- Predicts stock prices based on historical data.
- Implements a RNN or deep learning LSTM model for time series forecasting.

---

## Installation
Provide clear instructions to set up the project:
1. Clone the repository:
    git clone https://github.com/Lorant7/airbnb_stock_prediction.git
2. Download data:
    python load.py
3. Train model:
    python train_and_val.py
4. Test model:
    python test.py
