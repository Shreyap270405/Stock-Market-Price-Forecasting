# predict.py
import numpy as np
import pandas as pd

def predict_next(model, data, scaler, look_back=60):
    """
    Predicts the next closing stock price using the trained LSTM model.
    
    Args:
        model: Trained LSTM model.
        data (pd.DataFrame): DataFrame containing a 'Close' column.
        scaler: MinMaxScaler used during training.
        look_back (int): Number of previous days to use for prediction.
    
    Returns:
        float: Predicted next day's stock closing price.
    """
    if 'Close' not in data.columns:
        raise ValueError("Input DataFrame must contain a 'Close' column")

    # Prepare the last 60 closing prices
    last_60_days = data['Close'].values[-look_back:].reshape(-1, 1)
    scaled_last_60 = scaler.transform(last_60_days)

    # Reshape for LSTM input
    X_test = np.array([scaled_last_60])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Predict
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    return float(predicted_price[0][0])


def predict_horizon(model, data, scaler, look_back=60, days=30):
    """
    Predict multiple future closing prices by iteratively feeding predictions back as inputs.

    Args:
        model: Trained LSTM model.
        data (pd.DataFrame): DataFrame containing a 'Close' column.
        scaler: MinMaxScaler used during training (expects same feature shape).
        look_back (int): Number of previous days used for prediction.
        days (int): Number of future days to predict.

    Returns:
        list[float]: Predicted prices for the next `days` days in chronological order.
    """
    # Validate
    if 'Close' not in data.columns:
        raise ValueError("Input DataFrame must contain a 'Close' column")

    # Prepare the last look_back closing prices
    closes = pd.to_numeric(data['Close'], errors='coerce').dropna()
    if len(closes) < look_back:
        raise ValueError(f"Not enough history to create a sequence of length {look_back} (have {len(closes)}).")

    last_vals = closes.values[-look_back:].reshape(-1, 1)
    scaled_window = scaler.transform(last_vals)

    preds_scaled = []
    current_window = scaled_window.copy()

    for _ in range(days):
        X = np.array([current_window])  # shape (1, look_back, 1)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        pred_scaled = model.predict(X)
        # pred_scaled shape (1, 1)
        preds_scaled.append(pred_scaled[0][0])
        # append prediction to current window and drop first
        new_row = np.array([[pred_scaled[0][0]]])
        current_window = np.vstack([current_window[1:], new_row])

    # Inverse transform predictions
    preds_scaled_arr = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled_arr)
    return [float(x[0]) for x in preds]
