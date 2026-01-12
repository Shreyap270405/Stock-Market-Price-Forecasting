# data_fetch.py
import yfinance as yf
import pandas as pd
import os

def fetch_ticker(ticker="AAPL", period="5y", save_csv=False):
    """
    Fetch stock data using Yahoo Finance for a given ticker and time period.

    Args:
        ticker (str): Stock ticker symbol, e.g., "AAPL"
        period (str): Duration like "1y", "5y", "max"
        save_csv (bool): Whether to save the data to a CSV file

    Returns:
        pandas.DataFrame: Stock data with Date, Open, High, Low, Close, Adj Close, Volume
    """
    try:
        df = yf.download(ticker, period=period)
        df.reset_index(inplace=True)

        if save_csv:
            os.makedirs("data", exist_ok=True)
            csv_path = f"data/{ticker}_data.csv"
            df.to_csv(csv_path, index=False)
            print(f"✅ Data saved to {csv_path}")

        return df

    except Exception as e:
        print(f"⚠️ Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
