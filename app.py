import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from polygon import RESTClient
from fredapi import Fred
import yfinance as yf
import os
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

# ==========================================
# 1. API KEY INITIALIZATION (Fail-Safe)
# ==========================================
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

if not POLYGON_API_KEY:
    try:
        POLYGON_API_KEY = st.secrets.get("POLYGON_API_KEY")
        FRED_API_KEY = st.secrets.get("FRED_API_KEY")
    except:
        pass

if not POLYGON_API_KEY or not FRED_API_KEY:
    st.set_page_config(page_title="API Setup Required", page_icon="🔑")
    st.error("### 🔑 API Keys Missing")
    st.info("Please add POLYGON_API_KEY and FRED_API_KEY to your Space Secrets in Settings.")
    st.stop()

polygon_client = RESTClient(POLYGON_API_KEY)
fred = Fred(api_key=FRED_API_KEY)

# ==========================================
# 2. LSTM MODEL ARCHITECTURE
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# 3. DATA FETCHING (Hybrid Engine)
# ==========================================
@st.cache_data(ttl=3600)
def get_hybrid_data(ticker_list, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    end_str = end_date.strftime('%Y-%m-%d')
    start_str = start_date.strftime('%Y-%m-%d')
    combined_data = pd.DataFrame()
    
    progress_bar = st.progress(0)
    for i, ticker in enumerate(ticker_list):
        success = False
        # Try Polygon
        try:
            aggs = polygon_client.get_aggs(ticker, 1, "day", start_str, end_str)
            df = pd.DataFrame(aggs)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            combined_data[ticker] = df['close']
            success = True
        except Exception:
            pass # Failover to yfinance
        
        # Fallback to yfinance
        if not success:
            try:
                yf_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not yf_data.empty:
                    if isinstance(yf_data.columns, pd.MultiIndex):
                        combined_data[ticker] = yf_data['Close'][ticker]
                    else:
                        combined_data[ticker] = yf_data['Close']
            except Exception:
                st.error(f"Failed to fetch {ticker}")

        progress_bar.progress((i + 1) / len(ticker_list))
    return combined_data

# ==========================================
# 4. APP UI
# ==========================================
st.set_page_config(page_title="Fixed Income ETF Maximizer", layout="wide")
st.title("📈 Fixed Income ETF Return Maximizer")

# Sidebar
st.sidebar.header("Parameters")
lookback_days = st.sidebar.slider("Historical Lookback (Days)", 365, 3650, 1825)
fixed_income_etfs = ["VCLT", "TLT", "TBT", "MBB", "VNQ", "HYG", "SLV", "GLD", "PFF"]
selected_tickers = st.sidebar.multiselect("Select ETFs", fixed_income_etfs, default=fixed_income_etfs)

if selected_tickers:
    data = get_hybrid_data(selected_tickers, lookback_days)
    
    if not data.empty:
        # Charting
        st.subheader("Historical Performance")
        fig = go.Figure()
        for t in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[t], name=t))
        st.plotly_chart(fig, width='stretch')

        # Model Section
        st.divider()
        st.header("🤖 Model Training & Prediction")
        
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("Training Epochs", 1, 100, 20)
            window_size = st.slider("Window Size (Days)", 10, 60, 30)
        
        if st.button("🚀 Train LSTM Model"):
            with st.status("Training Neural Network...", expanded=True) as status:
                # Preprocessing
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data.dropna())
                
                X, y = [], []
                for i in range(len(scaled_data) - window_size):
                    X.append(scaled_data[i:i+window_size])
                    y.append(scaled_data[i+window_size])
                
                X_train = torch.FloatTensor(np.array(X))
                y_train = torch.FloatTensor(np.array(y))

                # Initialize Model
                model = LSTMModel(input_size=len(selected_tickers), hidden_size=64, num_layers=2, output_size=len(selected_tickers))
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                # Train
                for epoch in range(epochs):
                    model.train()
                    outputs = model(X_train)
                    optimizer.zero_grad()
                    loss = criterion(outputs, y_train)
                    loss.backward()
                    optimizer.step()
                    if (epoch+1) % 5 == 0:
                        status.write(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

                status.update(label="Training Complete!", state="complete")
                
                # Predict
                model.eval()
                last_window = torch.FloatTensor(scaled_data[-window_size:].reshape(1, window_size, -1))
                prediction = model(last_window).detach().numpy()
                predicted_prices = scaler.inverse_transform(prediction)
                
                st.subheader("Next Day Forecast")
                pred_df = pd.DataFrame(predicted_prices, columns=selected_tickers, index=["Predicted Price"])
                st.table(pred_df)
    else:
        st.error("No data found for the selected tickers.")
else:
    st.info("Select ETFs in the sidebar to begin.")
