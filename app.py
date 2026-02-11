import os
import streamlit as st

# 1. Try to get keys from Environment Variables (Hugging Face style)
# 2. Fall back to Streamlit secrets (Local development style)
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY") or st.secrets.get("POLYGON_API_KEY")
FRED_API_KEY = os.environ.get("FRED_API_KEY") or st.secrets.get("FRED_API_KEY")

# Safety Check: If both are missing, show a helpful message instead of crashing
if not POLYGON_API_KEY:
    st.error("🔑 POLYGON_API_KEY not found. Please check your Hugging Face Space Secrets.")
    st.stop()
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from polygon import RESTClient

# ────────────────────────────────────────────────
# 1. Configuration & API Setup
# ────────────────────────────────────────────────
st.set_page_config(page_title="Trading Engine", layout="wide")

# Fetch API Key from HF Secrets or Local Streamlit Secrets
POLYGON_API_KEY = st.secrets.get("POLYGON_API_KEY") or os.environ.get("POLYGON_API_KEY")

ETFS = ['TLT', 'VCLT', 'TBT', 'VNQ', 'SLV', 'GLD', 'PFF', 'MBB']
SIGNAL_TICKERS = ['IEF', 'SHY', 'HYG'] 
ALL_TICKERS = ETFS + SIGNAL_TICKERS

device = torch.device("cpu") # Force CPU for HF stability

# ────────────────────────────────────────────────
# 2. Data Fetching
# ────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_data(start_date, end_date):
    if not POLYGON_API_KEY:
        st.error("Missing Polygon API Key! Please add it to your Space Secrets.")
        return pd.DataFrame()
    
    client = RESTClient(api_key=POLYGON_API_KEY)
    data = {}
    
    for ticker in ALL_TICKERS:
        try:
            aggs = client.get_aggs(ticker, 1, "day", start_date, end_date)
            # Convert Polygon objects to a list of dicts for Pandas
            rows = [{"date": pd.to_datetime(a.timestamp, unit='ms').date(), ticker: a.close} for a in aggs]
            df = pd.DataFrame(rows).set_index('date')
            data[ticker] = df.astype(float)
        except Exception as e:
            st.warning(f"Could not fetch {ticker}: {e}")
    
    if not data: return pd.DataFrame()
    combined = pd.concat(data.values(), axis=1).sort_index().ffill().bfill()
    combined.index = pd.to_datetime(combined.index)
    return combined

# ────────────────────────────────────────────────
# 3. Model Architecture & Training
# ────────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=8):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]) # Predict based on last time step

def engineer_features(df):
    df = df.copy()
    # Adding basic technical indicators as features
    for col in ALL_TICKERS:
        if col in df.columns:
            df[f'{col}_ret'] = df[col].pct_change()
            df[f'{col}_ma5'] = df[col].rolling(5).mean()
    return df.dropna()

# ────────────────────────────────────────────────
# 4. Main UI Logic
# ────────────────────────────────────────────────
st.title("Fixed Income LSTM Trading Engine")

# Date range logic
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=1000)).strftime("%Y-%m-%d")

if st.sidebar.button("Start Analysis"):
    with st.spinner("Fetching data and training model..."):
        df_raw = fetch_data(start_date, end_date)
        
        if not df_raw.empty:
            df_eng = engineer_features(df_raw)
            st.write("Latest Data Snapshot", df_eng.tail(5))
            
            # Simple visualization
            fig = go.Figure()
            for etf in ETFS:
                if etf in df_raw.columns:
                    fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw[etf], name=etf))
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("Analysis Complete. Model is ready for backtesting.")
        else:
            st.error("No data found. Check your API limits.")
