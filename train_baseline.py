# train_baseline.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import argparse
from preprocess import create_features

def train_baseline(csv_path, model_out='models/rf_baseline.joblib'):
    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    dff = create_features(df)
    # features: last day's Close, MA7, MA21, Volatility
    dff = dff = dff.dropna()
    dff['PrevClose'] = dff['Close'].shift(1)
    dff = dff.dropna()
    feature_cols = ['PrevClose','MA7','MA21','Volatility']
    X = dff[feature_cols]
    y = dff['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print("Baseline RMSE:", rmse)
    joblib.dump(model, model_out)
    print("Saved baseline model to", model_out)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/AAPL_5y_1d.csv")
    args = parser.parse_args()
    train_baseline(args.csv)
