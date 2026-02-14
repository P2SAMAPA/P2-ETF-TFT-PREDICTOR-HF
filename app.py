import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from huggingface_hub import hf_hub_download, HfApi
import os
import gc
from datetime import datetime, timedelta

# --- 1. PERSISTENCE ENGINE ---
DATASET_REPO_ID = "P2SAMAPA/my-etf-data"
FILENAME = "historical_cache.parquet"
HF_TOKEN = st.secrets.get("HF_TOKEN")

def sync_data(assets, start_year):
    """Handles the pull-update-push cycle for zero-amnesia data."""
    df = None
    # Attempt to pull existing data from your private dataset
    if HF_TOKEN:
        try:
            path = hf_hub_download(repo_id=DATASET_REPO_ID, filename=FILENAME, repo_type="dataset", token=HF_TOKEN)
            df = pd.read_parquet(path)
            st.sidebar.success("✅ Cloud Cache Loaded")
        except Exception:
            st.sidebar.warning("⚠️ No cloud file found. Fetching full history...")

    # Fetch from Yahoo if cache is missing or doesn't go back far enough for the anchor
    target_start = f"{start_year}-01-01"
    if df is None or df.index.min() > pd.to_datetime(target_start):
        df = yf.download(assets, start=target_start, progress=False)['Close']
    else:
        # Incremental update for today's data
        last_date = df.index.max()
        if last_date.date() < datetime.now().date():
            new_data = yf.download(assets, start=(last_date + timedelta(days=1)).strftime('%Y-%m-%d'), progress=False)['Close']
            if not new_data.empty:
                df = pd.concat([df, new_data]).drop_duplicates().sort_index()

    # Push updated version back to P2SAMAPA/my-etf-data
    if HF_TOKEN:
        df.to_parquet(FILENAME)
        try:
            HfApi().upload_file(path_or_fileobj=FILENAME, path_in_repo=FILENAME, repo_id=DATASET_REPO_ID, repo_type="dataset", token=HF_TOKEN)
        except Exception as e:
            st.sidebar.error(f"Push Failed: {e}")
    return df

# --- 2. ARCHITECTURE ---
class MomentumTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, num_heads=8, seq_len=60):
        super(MomentumTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        # 0.3 Dropout for regime stability
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=0.3, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)
        self.decoder = nn.Linear(d_model, 6) 

    def forward(self, x):
        x = self.input_projection(x) + self.pos_encoder
        x = self.transformer_encoder(x)
        return self.decoder(x[:, -1, :])

# --- 3. TRAINING ENGINE ---
@st.cache_resource(ttl="30m")
def train_engine(start_year, tx_cost):
    gc.collect()
    etfs = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
    lookback = 60 # Hard-coded per 60-day finding
    
    data = sync_data(etfs, start_year)
    returns_df = data.ffill().pct_change().fillna(0)
    returns_df['CASH'] = 0.0001
    
    # Feature Engineering
    for asset in etfs + ['CASH']:
        returns_df[f'{asset}_ROC_10'] = returns_df[asset].pct_change(10)
    
    features_df = returns_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    target_df = returns_df[etfs + ['CASH']].rolling(3).sum().shift(-3).dropna()
    
    # Anchor Filter & 80:20 Split
    filtered_features = features_df[features_df.index >= pd.to_datetime(f"{start_year}-01-01")]
    split_idx = int(len(filtered_features) * 0.8)
    
    train_feat = filtered_features.iloc[:split_idx]
    oos_feat = filtered_features.iloc[split_idx:]
    
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_feat.astype(np.float64))
    scaled_oos = scaler.transform(oos_feat.astype(np.float64))
    
    X_train = torch.FloatTensor(np.array([scaled_train[i:i+lookback] for i in range(len(scaled_train)-lookback)]))
    y_train = torch.FloatTensor(target_df.loc[train_feat.index].iloc[lookback:].values)

    model = MomentumTransformer(input_dim=features_df.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    model.train()
    for _ in range(100):
        optimizer.zero_grad()
        output = model(X_train)
        loss = nn.MSELoss()(output, y_train[:len(output)])
        loss.backward(); optimizer.step()
        
    return {"model": model, "scaler": scaler, "returns": returns_df[etfs+['CASH']], 
            "oos_features": scaled_oos, "oos_dates": oos_feat.index, "lookback": lookback}

# --- 4. DASHBOARD ---
st.set_page_config(page_title="Alpha V6.2", layout="wide")

with st.sidebar:
    st.header("Strategy Controls")
    anchor = st.slider("Year Anchor (Pool Start)", 2008, 2023, 2019)
    tx = st.slider("TX Cost (BPS)", 0, 50, 15)
    with st.spinner("Processing Regime..."):
        engine = train_engine(anchor, tx)
    st.info(f"OOS Window: {len(engine['oos_dates'])} Trading Days")

# Inference
assets = ["TLT", "TBT", "VNQ", "SLV", "GLD", "CASH"]
lb = engine["lookback"]
engine["model"].eval()
picks = []
for i in range(len(engine["oos_features"])-lb):
    with torch.no_grad():
        pred = engine["model"](torch.FloatTensor(engine["oos_features"][i:i+lb]).unsqueeze(0)).numpy()[0]
    picks.append(assets[np.argmax(pred)])

# Performance Calc
res_df = pd.DataFrame({"Pick": picks, "Return": [engine["returns"].loc[engine["oos_dates"][i+lb], p] for i, p in enumerate(picks)]}, index=engine["oos_dates"][lb:])
wealth = (1 + (res_df["Return"] - tx/10000)).cumprod()

st.title("Alpha Maximizer V6.2")
c1, c2, c3 = st.columns(3)
c1.metric("Ann. Return", f"{(res_df['Return'].mean()*252)*100:.1f}%")
c2.metric("Sharpe", f"{(res_df['Return'].mean()/res_df['Return'].std())*np.sqrt(252):.2f}")
c3.metric("Current Signal", picks[-1])

st.line_chart(wealth)
st.subheader("Recent Strategy Log")
st.table(res_df.tail(10))
