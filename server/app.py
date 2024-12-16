from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import datetime
import numpy as np

app = Flask(__name__)

def fetch_data(ticker, start_date, end_date):
    # Fetch historical stock data
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def predict_prices(df):
    # Predict high, low, and average close prices for the next 5 days.
    predictions = []
    return predictions

def recommend_trading_strategy(predictions, nvda_open, nvdq_open):
    # Recommend trading strategies based on predictions.
    strategy = []
    for day in predictions:
        if day['high'] > nvda_open * 1.02:  # NVDA is predicted to go up by 2%
            strategy.append("BULLISH")
        elif day['low'] < nvda_open * 0.98:  # NVDA might drop by 2%
            strategy.append("BEARISH")
        else:
            strategy.append("IDLE")
    return strategy

def paper_trade(strategy, nvda_open, nvdq_open, nvda_shares, nvdq_shares):
    # Simulate paper trading for 5 business days based on strategy.
    for action in strategy:
        if action == "BULLISH":
            # Swap all NVDQ shares for NVDA shares
            nvda_shares += nvdq_shares * (nvdq_open / nvda_open)
            nvdq_shares = 0
        elif action == "BEARISH":
            # Swap all NVDA shares for NVDQ shares
            nvdq_shares += nvda_shares * (nvda_open / nvdq_open)
            nvda_shares = 0
    return nvda_shares, nvdq_shares

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            print("Inside POST")
            # Get current date from user input
            user_date = request.form["date"]
            user_date = datetime.datetime.strptime(user_date, "%Y-%m-%d")
            start_date = user_date - datetime.timedelta(days=30)
            end_date = user_date

            # Fetch stock price data for NVDA and NVDQ
            nvda_data = fetch_data("NVDA", start_date, end_date)
            nvdq_data = fetch_data("NVDQ", start_date, end_date)

            if nvda_data is None or nvdq_data is None:
                return jsonify({"error": "Failed to fetch the data"})

            # Latest open prices
            nvda_open = nvda_data['Open'].iloc[-1]
            nvdq_open = nvdq_data['Open'].iloc[-1]

            # Predict prices for next 5 days
            predictions = predict_prices(nvda_data)

            # Recommend trading strategies
            strategy = recommend_trading_strategy(predictions, nvda_open, nvdq_open)

            # Paper trading
            nvda_shares, nvdq_shares = paper_trade(strategy, nvda_open, nvdq_open, 10000, 100000)

            # Net portfolio value after 5 days
            final_value = nvda_shares * nvda_open + nvdq_shares * nvdq_open

            result = {
                "predictions": predictions,
                "strategy": strategy,
                "final_value": round(final_value, 2)
            }
            return jsonify(result)

        except Exception as e:
            return jsonify({"error": str(e)})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
