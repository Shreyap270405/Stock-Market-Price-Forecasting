# app_streamlit.py
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd


from data_fetch import fetch_ticker

import os
from train_lstm import train
from predict import predict_next, predict_horizon
import matplotlib.pyplot as plt
import joblib
from utils import load_model

st.title("Stock Price Forecasting Demo")

ticker = st.text_input("Ticker", value="AAPL")
period = st.selectbox("Period", ["1y","2y","5y","10y"], index=2)
if st.button("Fetch & Show History"):
    df = fetch_ticker(ticker=ticker, period=period, save_csv=False)
    st.line_chart(df['Close'])
    # save locally for other scripts
    os.makedirs('data', exist_ok=True)
    df.to_csv(f"data/{ticker}_{period}_1d.csv")
    st.success("Data fetched and saved to data/")

st.markdown("### Train & Predict (quick demo)")
if st.button("Train LSTM (quick)"):
    csv_path = f"data/{ticker}_{period}_1d.csv"
    if not os.path.exists(csv_path):
        st.error("Fetch history first.")
    else:
        with st.spinner("Training (short)..."):
            model, scaler, hist = train(csv_path, seq_len=60, epochs=5)  # short epochs for demo
            # Save model and scaler for later predictions
            os.makedirs('models', exist_ok=True)
            try:
                joblib.dump(scaler, 'models/scaler.save')
            except Exception:
                # fallback: try to use pickle via joblib anyway
                joblib.dump(scaler, 'models/scaler.save')
            try:
                # Prefer project's utils.save_model if available, otherwise keras save inside train
                from utils import save_model
                save_model(model, path='models/stock_model.h5')
            except Exception:
                # best-effort: attempt to use keras save
                try:
                    model.save('models/stock_model.h5')
                except Exception:
                    pass
            # keep in session so predict button can use in the same run
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.success("Training finished (demo). Model and scaler saved to models/")

st.markdown("### Predict Future")
horizon = st.selectbox("Horizon", ["Next day", "Full month (30 days)", "Full year (252 days)", "Custom (days)"], index=0)
custom_days = None
if horizon == "Custom (days)":
    custom_days = st.number_input("Days to predict", min_value=1, max_value=1000, value=30)
# For 'Next day' remove the checkbox and always show the small next-day plot.
# For other horizons keep the checkbox to let users toggle the graph.
if horizon == "Next day":
    show_graph = True
else:
    # Option to toggle plotting of prediction graph for multi-day horizons
    show_graph = st.checkbox("Show prediction graph", value=True)

if st.button("Predict"):
    csv_path = f"data/{ticker}_{period}_1d.csv"
    if not os.path.exists(csv_path):
        st.error("Fetch history first.")
    else:
        # Load model/scaler from session or disk, then load data and predict
        model = st.session_state.get('model')
        scaler = st.session_state.get('scaler')
        if model is None:
            model = load_model('models/stock_model.h5')
        if scaler is None:
            try:
                scaler = joblib.load('models/scaler.save')
            except Exception:
                scaler = None

        if model is None or scaler is None:
            st.error('No trained model/scaler found. Train the model first (or ensure models/ contains saved artifacts).')
        else:
            # read CSV and parse Date if available
            try:
                df = pd.read_csv(csv_path, parse_dates=['Date'], infer_datetime_format=True)
            except Exception:
                df = pd.read_csv(csv_path)

            # decide days
            if horizon == "Next day":
                days = 1
            elif horizon == "Full month (30 days)":
                days = 30
            elif horizon == "Full year (252 days)":
                days = 252
            else:
                days = int(custom_days or 30)

            try:
                if days == 1:
                    price = predict_next(model, df, scaler)
                    st.write(f"Predicted next close price: {price:.2f}")
                else:
                    preds = predict_horizon(model, df, scaler, look_back=60, days=days)
                    # prepare dates for plotting if Date column exists
                    last_date = None
                    if 'Date' in df.columns:
                        try:
                            last_date = pd.to_datetime(df['Date'].iloc[-1])
                        except Exception:
                            last_date = None
                    if last_date is not None:
                        future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days)
                        preds_df = pd.DataFrame({'Date': future_dates, 'PredictedClose': preds})
                        preds_df = preds_df.set_index('Date')
                        # show table and optionally plot together with history
                        st.dataframe(preds_df)
                        if show_graph:
                            hist = df.copy()
                            if 'Date' in hist.columns:
                                try:
                                    hist['Date'] = pd.to_datetime(hist['Date'])
                                    hist = hist.set_index('Date')
                                except Exception:
                                    pass
                            # combine last portion of history with predictions for visibility
                            try:
                                to_plot = pd.concat([hist['Close'].dropna().iloc[-120:], preds_df['PredictedClose']])
                            except Exception:
                                to_plot = pd.concat([hist['Close'].dropna(), preds_df['PredictedClose']])
                            st.line_chart(to_plot)
                    else:
                        # no workable dates: show table with t+1..t+n and optionally plot
                        preds_df = pd.DataFrame({'t+{}'.format(i+1): [p] for i,p in enumerate(preds)})
                        st.dataframe(preds_df)
                        if show_graph:
                            st.line_chart(pd.Series(preds))
            except Exception as e:
                st.error(f"Prediction failed: {e}")
