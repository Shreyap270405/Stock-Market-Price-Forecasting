# Stock Market Price Forecasting using LSTM

This project is an end-to-end **Stock Price Forecasting Web Application** built using **Python, TensorFlow (LSTM), and Streamlit**. It predicts future stock prices based on historical data using deep learning techniques and provides an interactive web interface for training and visualization.

---

## Features
- Upload historical stock price CSV files
- Train an LSTM-based deep learning model
- Visualize stock price trends
- Forecast future stock prices (next 30 days)
- Interactive and user-friendly Streamlit UI
- Modular and well-structured Python code

---

## Technologies Used
- **Programming Language:** Python  
- **Deep Learning:** TensorFlow / Keras (LSTM)  
- **Web Framework:** Streamlit  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib  
- **Machine Learning Utilities:** Scikit-learn  

---

## Project Structure
stock-price-forecasting/
│
├── app_streamlit.py # Streamlit web application
├── train_lstm.py # LSTM model training logic
├── predict.py # Future price prediction logic
├── data_fetch.py # Stock data fetching
├── utils.py # Utility functions
├── requirements.txt # Dependencies
├── sample_data.csv # Sample dataset
└── .venv/ # Virtual environment

yaml
Copy code

---

## Dataset
- Historical stock price data in CSV format
- Dataset must include a **Close** price column
- Data sources: Yahoo Finance, Kaggle

---

## Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/stock-price-forecasting.git
cd stock-price-forecasting
2️⃣ Create & Activate Virtual Environment
bash
Copy code
python -m venv .venv
.\.venv\Scripts\activate
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
Or install manually:

bash
Copy code
pip install streamlit tensorflow scikit-learn pandas numpy matplotlib yfinance
▶️ How to Run the Project
bash
Copy code
streamlit run app_streamlit.py
Open your browser and visit:

arduino
Copy code
http://localhost:8501
📈 How It Works
User uploads historical stock price data

Data is normalized and converted into time-series sequences

LSTM model is trained using past stock prices

Model predicts future stock price trends

Results are visualized using interactive charts

⚠️ Disclaimer
This project is created for educational purposes only.
Stock market predictions are probabilistic and should not be used for real-world trading decisions.

👩‍💻 Author
Shreya Pandey
B.E. Computer Science | Data Science & Machine Learning Enthusiast

⭐ Acknowledgements
TensorFlow & Keras Documentation

Streamlit Community

Yahoo Finance
