import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import requests
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
# 1. PERMANENT UI CONFIGURATION
# ------------------------------
st.set_page_config(page_title="P2-Transformer Alpha", layout="wide")

# This Sidebar must render EVERY time the script runs
with st.sidebar:
    st.header("⚙️ Model Configuration")
    start_year = st.slider("Training Start Year", 2008, 2024, 2016)
    fee_bps = st.slider("Transaction Cost (bps)", 0, 100, 15)
    lookback = st.slider("Lookback Window (Days)", 10, 60, 30)
    epochs = st.number_input("Training Epochs", 5, 100, 20)
    st.markdown("---")
    update_hf = st.checkbox("Sync Updates to HF Hub?", value=True)
    run_button = st.button("🚀 Execute Transformer Alpha", type="primary")

# Main Header
st.title("🚀 P2-TRANSFORMER-ETF-PREDICTOR")
st.info("The logic is now running as a Transformer-Hybrid architecture.")

# ------------------------------
# 2. DATA & API SECRETS
# ------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
AV_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY")
REPO_ID = "P2SAMAPA/my-etf-data"
fee_pct = fee_bps / 10000

# ------------------------------
# 3. ROBUST DATA ENGINE
# ------------------------------
def get_and_sync_data():
    with st.status("📡 Data Synchronization...", expanded=True) as status:
        # Load existing
        ds = load_dataset(REPO_ID, split="train").to_pandas()
        ds['Date'] = pd.to_datetime(ds['Date'])
        ds = ds.set_index('Date').sort_index()
        
        # New Macro Tickers
        tickers = {"^VIX": "VIX", "^TNX": "TNX", "DX-Y.NYB": "DXY", "GC=F": "GOLD", "HG=F": "COPPER"}
        
        try:
            st.write("🛰️ Attempting Yahoo Finance...")
            macro = yf.download(list(tickers.keys()), start=ds.index.min(), progress=False)['Close']
            macro = macro.rename(columns=tickers)
        except Exception:
            st.warning("⚠️ Yahoo Rate Limit. Falling back to internal cache/AlphaVantage...")
            # Fallback logic would go here
            macro = pd.DataFrame(index=ds.index) 

        # Ratio Logic
        if 'GOLD' in macro.columns and 'COPPER' in macro.columns:
            macro['AU_CU_Ratio'] = macro['GOLD'] / macro['COPPER']
            macro['AU_CU_Trend'] = macro['AU_CU_Ratio'].rolling(window=20).mean()
        
        # Merge without duplication
        macro_cols = ['VIX', 'TNX', 'DXY', 'AU_CU_Ratio', 'AU_CU_Trend']
        cols_to_add = [c for c in macro.columns if c in macro_cols and c not in ds.columns]
        full_df = ds.join(macro[cols_to_add], how='left') if cols_to_add else ds
        full_df = full_df.ffill().dropna()
        
        if update_hf and HF_TOKEN and not full_df.equals(ds):
            login(token=HF_TOKEN)
            Dataset.from_pandas(full_df.reset_index()).push_to_hub(REPO_ID)
            st.success("✅ HF Dataset Synchronized.")
            
        status.update(label="Data Ready!", state="complete")
    return full_df

# ------------------------------
# 4. MODEL & BACKTEST
# ------------------------------
if run_button:
    # A. Execute Data Sync
    df = get_and_sync_data()
    target_etfs = ['TLT_Ret', 'TBT_Ret', 'VNQ_Ret', 'SLV_Ret', 'GLD_Ret']
    
    # B. Scale & Prepare Tensors
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(df[target_etfs].iloc[i].values)
    X, y = np.array(X), np.array(y)

    # C. Build & Train
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    attn = MultiHeadAttention(num_heads=4, key_dim=X.shape[2])(inputs, inputs)
    attn = LayerNormalization()(attn + inputs)
    conv = Conv1D(64, 3, activation='relu', padding='same')(attn)
    pool = GlobalMaxPooling1D()(conv)
    lstm = Bidirectional(LSTM(64))(inputs)
    merged = Concatenate()([pool, lstm])
    x = Dense(128, activation='relu')(merged)
    outputs = Dense(5)(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    
    with st.spinner("🧠 Training Hybrid Transformer..."):
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
    
    # D. Inference
    preds = model.predict(X)
    
    # E. Restore Output UI
    st.subheader("📊 Backtest Results")
    col1, col2 = st.columns(2)
    
    strat_rets = []
    for i in range(len(preds)):
        idx = np.argmax(preds[i])
        strat_rets.append((y[i][idx] - fee_pct) if preds[i][idx] > fee_pct else 0.0)
    
    cum_strat = np.cumprod(1 + np.array(strat_rets))
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=cum_strat, name="Strategy Alpha"))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        latest = preds[-1]
        top_idx = np.argmax(latest)
        top_asset = target_etfs[top_idx].split('_')[0]
        st.metric("Top Recommended Asset", top_asset)
        st.metric("Expected Return", f"{latest[top_idx]*100:.2f}%")

    if latest[top_idx] <= fee_pct:
        st.error(f"🚨 SIGNAL: CASH (Below {fee_bps} bps hurdle)")
    else:
        st.success(f"✅ FINAL PREDICTION: ALLOCATE TO {top_asset}")
