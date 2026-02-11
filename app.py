import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import os
import time

# ==========================================
# 1. MODELS (VAE + LSTM)
# ==========================================
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=3):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, latent_dim * 2))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 16), nn.ReLU(), nn.Linear(16, input_dim), nn.Sigmoid())

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decoder(z)

class SuperLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super(SuperLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, 64, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==========================================
# 2. AUTO-TRAINING ENGINE (3-Day TTL)
# ==========================================
@st.cache_resource(ttl="3d", show_spinner="Training Super-Engine (Refresh Every 3 Days)...")
def train_super_engine(start_year, etf_list):
    # Fetch Data
    start_date = f"{start_year}-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Unified Data Fetch
    prices = yf.download(etf_list, start=start_date, end=end_date)['Close']
    
    # Macro Features
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    macro = pd.DataFrame(index=prices.index)
    macro['10Y2Y'] = fred.get_series('T10Y2Y', start_date, end_date)
    macro['10Y5Y'] = fred.get_series('T10Y5Y', start_date, end_date)
    
    m_tickers = {"^MOVE": "MOVE", "^SKEW": "SKEW", "^PCCR": "PCC", "GC=F": "Gold", "HG=F": "Copper"}
    m_raw = yf.download(list(m_tickers.keys()), start=start_date, end=end_date)['Close'].rename(columns=m_tickers)
    m_raw['Au_Cu'] = m_raw['Gold'] / m_raw['Copper']
    
    full_df = pd.concat([prices, macro, m_raw[['MOVE', 'SKEW', 'PCC', 'Au_Cu']]], axis=1).ffill().dropna()
    
    # Scale and Denoise
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(full_df)
    
    # Train Ensemble
    # [VAE Denoising logic here...]
    # [LSTM Training logic here...]
    
    # Return everything needed for inference
    return {"model": model, "scaler": scaler, "data": full_df, "trained_at": datetime.now()}

# ==========================================
# 3. UI DASHBOARD
# ==========================================
st.set_page_config(page_title="Institutional Alpha Engine", layout="wide")
st.title("🏛️ Institutional Alpha Engine")

# Controls
st.sidebar.header("Regime & Strategy")
regime_year = st.sidebar.select_slider("Data Anchor (History)", options=[2008, 2015, 2019, 2021])
tx_cost = st.sidebar.slider("Transaction Cost (bps)", 0, 100, 15)

etf_universe = ["VCLT", "TLT", "TBT", "MBB", "VNQ", "HYG", "SLV", "GLD", "PFF"]

# Execution
engine = train_super_engine(regime_year, etf_universe)
st.sidebar.caption(f"Engine Last Trained: {engine['trained_at'].strftime('%Y-%m-%d %H:%M')}")

# Optimization Logic
# (Uses the trained engine to deliver the 1D, 3D, 5D table as per previous code)
