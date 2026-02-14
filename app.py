import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import requests
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional, 
    Conv1D, GlobalMaxPooling1D, Concatenate, MultiHeadAttention, LayerNormalization
)
from sklearn.preprocessing import MinMaxScaler
from datasets import load_dataset, Dataset
from huggingface_hub import login
import plotly.graph_objects as go

# ------------------------------
# 1. SETUP & SECRETS
# ------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
AV_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY") # Add this to Space Secrets!
REPO_ID = "P2SAMAPA/my-etf-data"

st.set_page_config(page_title="P2-Transformer Pro", layout="wide")
st.title("🚀 P2-TRANSFORMER-ETF-PREDICTOR")

with st.sidebar:
    st.header("⚙️ Settings")
    fee_bps = st.slider("Transaction Cost (bps)", 0, 100, 15)
    fee_pct = fee_bps / 10000
    update_hf = st.checkbox("Sync to HF Hub?", value=True)
    run_button = st.button("🔥 Run Pipeline", type="primary")

# ------------------------------
# 2. FAILOVER DATA ENGINE
# ------------------------------
def fetch_alpha_vantage(symbol):
    """Fallback fetcher for AlphaVantage."""
    if not AV_API_KEY:
        return None
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={AV_API_KEY}&outputsize=full'
    r = requests.get(url)
    data = r.json()
    if "Time Series (Daily)" in data:
        df = pd.DataFrame(data["Time Series (Daily)"]).T
        return df['4. close'].astype(float)
    return None

def get_and_sync_data():
    with st.status("📡 Data Syncing...", expanded=True) as status:
        # Load existing
        ds = load_dataset(REPO_ID, split="train").to_pandas()
        ds['Date'] = pd.to_datetime(ds['Date'])
        ds = ds.set_index('Date').sort_index()
        
        # Target New Columns
        macro_cols = ['VIX', 'TNX', 'DXY', 'AU_CU_Ratio', 'AU_CU_Trend']
        
        try:
            st.write("🛰️ Attempting Yahoo Finance...")
            tickers = {"^VIX": "VIX", "^TNX": "TNX", "DX-Y.NYB": "DXY", "GC=F": "GOLD", "HG=F": "COPPER"}
            macro = yf.download(list(tickers.keys()), start=ds.index.min(), progress=False)['Close']
            macro = macro.rename(columns=tickers)
        except Exception:
            st.warning("⚠️ Yahoo Rate Limit Hit! Falling back to AlphaVantage...")
            # Simple fallback for Gold/Copper and VIX
            vix = fetch_alpha_vantage("VIX")
            gold = fetch_alpha_vantage("GOLD")
            copper = fetch_alpha_vantage("COPPER")
            macro = pd.DataFrame({"VIX": vix, "GOLD": gold, "COPPER": copper})
            # Note: TNX/DXY might require specific AV Global IDs
            
        # Calculate Barometer
        if 'GOLD' in macro.columns and 'COPPER' in macro.columns:
            macro['AU_CU_Ratio'] = macro['GOLD'] / macro['COPPER']
            macro['AU_CU_Trend'] = macro['AU_CU_Ratio'].rolling(window=20).mean()
        
        # Merge carefully to avoid the "Overlap" error
        cols_to_add = [c for c in macro.columns if c in macro_cols and c not in ds.columns]
        if cols_to_add:
            full_df = ds.join(macro[cols_to_add], how='left')
        else:
            full_df = ds
            
        full_df = full_df.ffill().dropna()
        
        if update_hf and HF_TOKEN and not full_df.equals(ds):
            login(token=HF_TOKEN)
            Dataset.from_pandas(full_df.reset_index()).push_to_hub(REPO_ID)
            st.success("✅ Dataset Updated!")
            
        status.update(label="Data Ready!", state="complete")
    return full_df

# ------------------------------
# 3. TRANSFORMER CORE (The Fixed Version)
# ------------------------------
def build_transformer(input_shape):
    inputs = Input(shape=input_shape)
    # Multi-Head Attention for Macro Regime Filtering
    attn = MultiHeadAttention(num_heads=4, key_dim=input_shape[1])(inputs, inputs)
    attn = LayerNormalization(epsilon=1e-6)(attn + inputs)
    
    # Dual Head: CNN + LSTM
    conv = Conv1D(64, 3, activation='relu', padding='same')(attn)
    pool = GlobalMaxPooling1D()(conv)
    lstm = Bidirectional(LSTM(64))(inputs)
    
    # Fusion and Dense Output
    merged = Concatenate()([pool, lstm])
    x = Dense(128, activation='relu')(merged)
    x = Dropout(0.2)(x)
    outputs = Dense(5)(x) # Properly called on x
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# ------------------------------
# 4. RUNTIME
# ------------------------------
if run_button:
    df = get_and_sync_data()
    target_etfs = ['TLT_Ret', 'TBT_Ret', 'VNQ_Ret', 'SLV_Ret', 'GLD_Ret']
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    
    # Build Tensors (30-day lookback)
    X, y = [], []
    for i in range(30, len(scaled)):
        X.append(scaled[i-30:i])
        y.append(df[target_etfs].iloc[i].values)
    
    X, y = np.array(X), np.array(y)
    
    with st.spinner("🧠 Training Transformer Strategy..."):
        model = build_transformer((X.shape[1], X.shape[2]))
        model.fit(X, y, epochs=20, batch_size=32, verbose=0)
    
    # Backtest Result
    preds = model.predict(X[-100:])
    # Pick top asset; if < fee, CASH.
    # ... UI Chart logic follows ...
    st.success("Final Prediction: " + target_etfs[np.argmax(preds[-1])].split('_')[0])
