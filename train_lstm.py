# train_lstm.py
 
import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def train(csv_path=None, seq_len=60, epochs=10):
    """
    Train an LSTM model on stock data.

    Args:
        csv_path (str): Path to CSV file containing stock data.
        seq_len (int): Sequence length (number of past days used for prediction).
        epochs (int): Number of epochs for training.

    Returns:
        model: Trained LSTM model
        scaler: MinMaxScaler for inverse transformation
        history: Training history object
    """
    # (csv is read below with robust handling)

    # Load dataset
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        raise ValueError("csv_path is required")

    # If the CSV has MultiIndex columns (e.g., from yfinance when multiple tickers
    # or when columns were saved with multiple header rows), flatten them to single
    # level names like 'Close' or 'AAPL_Close'.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if str(x) != '']).strip("_") for col in df.columns]

    # Try to find the close-like column (case-insensitive)
    close_col = None
    for col in df.columns:
        if str(col).strip().lower() == 'close' or 'close' in str(col).strip().lower():
            close_col = col
            break
    if close_col is None:
        raise ValueError("CSV must contain a 'Close' column (or a column name containing 'close')")

    # Coerce to numeric and drop invalid rows (this will remove rows like the extra
    # header/data rows that can contain strings such as 'AAPL')
    close_series = pd.to_numeric(df[close_col], errors='coerce').dropna()
    if close_series.empty:
        raise ValueError(f"No numeric data found in column '{close_col}'")

    data = close_series.values.reshape(-1, 1)

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Scale only numeric data
    scaled_data = scaler.fit_transform(data)

    # Prepare training data (80% train, 20% test)
    train_len = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_len]

    X_train, y_train = [], []
    for i in range(seq_len, len(train_data)):
        X_train.append(train_data[i-seq_len:i, 0])
        y_train.append(train_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    early_stop = EarlyStopping(monitor='loss', patience=3)
    history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=1, callbacks=[early_stop])

    return model, scaler, history
