from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import datetime
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

def is_weekend(date):
    # Check if given date is a weekend
    return date.weekday() in [5, 6]

def get_next_business_day(date):
    # Get next business day from today's date
    next_day = date + datetime.timedelta(days=1)
    while is_weekend(next_day):
        next_day += datetime.timedelta(days=1)
    return next_day

def fetch_data(ticker, start_date, end_date):
    # Fetch historical stock data
    try:
        nvda = yf.Ticker(ticker)
        data = nvda.history(start=start_date, end=end_date)
        data = data.dropna()
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        data = data.tail(60)
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None
    
def scale_data(scaler, data):
    return pd.DataFrame(scaler.transform(data), columns=data.columns)

def create_sequence(data, sequence_length):
    sequences = []
    seq = data[0:0 + sequence_length]
    sequences.append(seq)
    return np.array(sequences)

def predict_prices(model, input_sequence, scaler):
    # Predict high, low, and average close prices for the next 5 days.
    pred_scaled = model.predict(input_sequence)
    pred_scaled = pred_scaled.reshape(1, 5, 4) 
    padded_predictions = np.concatenate([pred_scaled, np.zeros((pred_scaled.shape[0], pred_scaled.shape[1], 1))], axis=2)
    full_predictions_rescaled = scaler.inverse_transform(padded_predictions.reshape(padded_predictions.shape[1], padded_predictions.shape[2]))
    predictions_rescaled = full_predictions_rescaled[:, :4]
    return [
        {"open": day[0], "high": day[1], "low": day[2], "close": day[3]}
        for day in predictions_rescaled
    ]

def recommend_trading_strategy(predictions):
    # Recommend trading strategies based on predictions.
    strategy = []
    for i in range(len(predictions)-1):
        if predictions[i+1]['open'] > predictions[i]['open'] * 1.02:
            strategy.append("BULLISH")
        elif predictions[i+1]['open'] < predictions[i]['open'] * 0.98:
            strategy.append("BEARISH")
        else:
            strategy.append("IDLE")
    if predictions[4]['close'] > predictions[4]['open'] * 1.02:
            strategy.append("BULLISH")
    elif predictions[4]['close'] < predictions[4]['open'] * 0.98:
        strategy.append("BEARISH")
    else:
        strategy.append("IDLE")
    return strategy


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            print("Inside POST")
            # Get current date from user input
            user_date = request.form["date"]
            user_date = datetime.datetime.strptime(user_date, "%Y-%m-%d")
            
            # Adjust start date to ensure there are enough business days
            start_date = user_date - datetime.timedelta(days=120)
            end_date = user_date
            print(start_date)
            print(end_date)
            
            # Fetch stock price data for NVDA and NVDQ
            nvda_data = fetch_data("NVDA", start_date, end_date)
            if nvda_data is None:
                return jsonify({"error": "Failed to fetch the data"})
            print("Data fetched")

            with open("nvda_scaler.pkl", "rb") as f:
                nvda_scaler = pickle.load(f)

            nvda_data_scaled = scale_data(nvda_scaler, nvda_data)
            print("Data scaled")

            input_sequence = create_sequence(nvda_data_scaled, 60)
            print("Input sequence created")

            nvda_model = load_model("nvda_model.keras")
            print("model loaded")
            
            # Predict prices for next 5 business days
            predictions = predict_prices(nvda_model, input_sequence, nvda_scaler)
            
            # Extract values for high, low, and close prices
            high_prices = [day['high'] for day in predictions]
            low_prices = [day['low'] for day in predictions]
            close_prices = [day['close'] for day in predictions]

            # Format predictions
            result = {
                "highest": round(max(high_prices), 2),
                "lowest": round(min(low_prices), 2),
                "average_close": round(sum(close_prices) / len(close_prices), 2)
            }
            print("price predicted")

            # start_date = get_next_business_day(user_date)
            # end_date = start_date + datetime.timedelta(days=12)
            # data = fetch_data("NVDA", start_date, end_date)
            # data = data.head(5)
            # print(data)
            print(predictions)

            # Recommend trading strategies
            strategy = recommend_trading_strategy(predictions)
            
            # Generate future business dates
            strategy_dates = []
            next_date = user_date
            for _ in range(5):
                next_date = get_next_business_day(next_date)
                strategy_dates.append(next_date.strftime("%Y-%m-%d"))
            
            strategy_list = [{"date": date, "action": action} for date, action in zip(strategy_dates, strategy)]
            print("strategy decided")

            final_result = {
                "highest": result["highest"],
                "lowest": result["lowest"],
                "average_close": result["average_close"],
                "strategy": strategy_list,
            }
            return jsonify(final_result)

        except Exception as e:
            return jsonify({"error": str(e)})

    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)