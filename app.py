import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import time
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
# 1. SETUP & AUTH
# ------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "P2SAMAPA/my-etf-data"

st.set_page_config(page_title="P2-Transformer Alpha", layout="wide")
st.title("🚀 P2-TRANSFORMER-ETF-PREDICTOR")
st.markdown("---")

# Restored Side-Bar Inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    start_year = st.slider("Start Year", 2008, 2024, 2016)
    fee_bps = st.slider("Transaction Cost (bps)", 0, 100, 15)
    fee_pct = fee_bps / 10000
    lookback = st.slider("Lookback Window", 10, 60, 30)
    epochs = st.number_input("Epochs", 10, 100, 30)
    update_hf = st.checkbox("Push Updates to HF Dataset?", value=True)
    run_button = st.button("🔥 Run Full Pipeline", type="primary")

# ------------------------------
# 2. DATA ENRICHMENT ENGINE
# ------------------------------
def get_and_sync_data():
    with st.status("📡 Fetching & Merging Macro Signals...", expanded=True) as status:
        # Load existing HF Data
        st.write("Loading existing dataset from Hugging Face...")
        ds = load_dataset(REPO_ID, split="train").to_pandas()
        ds['Date'] = pd.to_datetime(ds['Date'])
        ds = ds.set_index('Date').sort_index()
        
        # 5 New Signals: VIX, TNX, DXY, Gold, Copper
        st.write("Fetching VIX, TNX, DXY, Gold, Copper from Yahoo Finance...")
        tickers = {"^VIX": "VIX", "^TNX": "TNX", "DX-Y.NYB": "DXY", "GC=F": "GOLD", "HG=F": "COPPER"}
        macro = yf.download(list(tickers.keys()), start=ds.index.min())['Close']
        macro = macro.rename(columns=tickers)
        
        # Calculate Gold/Copper Ratio and its 20-day Trend
        st.write("Calculating Gold/Copper Barometer...")
        macro['AU_CU_Ratio'] = macro['GOLD'] / macro['COPPER']
        macro['AU_CU_Trend'] = macro['AU_CU_Ratio'].rolling(window=20).mean()
        
        # Join with existing columns (Returns, Vol, MA20, CPI, UNRATE)
        full_df = ds.join(macro[['VIX', 'TNX', 'DXY', 'AU_CU_Ratio', 'AU_CU_Trend']], how='left').ffill().dropna()
        
        # Push to Hub if requested
        if update_hf and HF_TOKEN:
            st.write("Pushing enriched data to Hugging Face...")
            login(token=HF_TOKEN)
            upload_df = full_df.reset_index()
            Dataset.from_pandas(upload_df).push_to_hub(REPO_ID)
            st.success("✅ Dataset Synced with 5 New Signals!")
        
        status.update(label="Data Synchronization Complete!", state="complete")
    return full_df

# ------------------------------
# 3. TRANSFORMER MODEL ARCHITECTURE
# ------------------------------
def build_transformer(input_shape):
    inputs = Input(shape=input_shape)
    
    # Multi-Head Attention for Macro Context Filtering
    # key_dim is the depth of the attention heads
    attn = MultiHeadAttention(num_heads=4, key_dim=input_shape[1])(inputs, inputs)
    attn = LayerNormalization(epsilon=1e-6)(attn + inputs)
    
    # Temporal CNN for short-term pattern recognition
    conv = Conv1D(64, kernel_size=3, activation='relu', padding='same')(attn)
    pool = GlobalMaxPooling1D()(conv)
    
    # Sequence memory via Bidirectional LSTM
    lstm = Bidirectional(LSTM(64))(inputs)
    
    # Concatenate local features (CNN), sequential features (LSTM), and attention
    merged = Concatenate()([pool, lstm])
    
    x = Dense(128, activation='relu')(merged)
    x = Dropout(0.2)(x)
    outputs = Dense(5) # Returns for TLT, TBT, VNQ, SLV, GLD
    
    return Model(inputs, outputs)

# ------------------------------
# 4. RUNTIME LOGIC
# ------------------------------
if run_button:
    # A. Data Prep
    df = get_and_sync_data()
    target_etfs = ['TLT_Ret', 'TBT_Ret', 'VNQ_Ret', 'SLV_Ret', 'GLD_Ret']
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    # B. Windowing for Transformer Input
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(df[target_etfs].iloc[i].values)
    
    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # C. Model Training
    with st.spinner(f"🧠 Training Hybrid Transformer for {epochs} epochs..."):
        model = build_transformer((X.shape[1], X.shape[2]))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    
    # D. Inference & Strategy
    preds = model.predict(X_test)
    
    strategy_returns = []
    for i in range(len(preds)):
        best_idx = np.argmax(preds[i])
        # Only trade if the best predicted return > transaction cost
        if preds[i][best_idx] > fee_pct:
            strategy_returns.append(y_test[i][best_idx] - fee_pct)
        else:
            strategy_returns.append(0.0) # Stay in CASH

    # E. UI Output
    st.subheader("📊 Performance Visualization")
    cum_strat = np.cumprod(1 + np.array(strategy_returns))
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=cum_strat, name="Transformer Alpha"))
    st.plotly_chart(fig, use_container_width=True)

    # FINAL RECOMMENDATION
    latest_pred = preds[-1]
    top_asset = target_etfs[np.argmax(latest_pred)].split('_')[0]
    
    if np.max(latest_pred) <= fee_pct:
        st.error(f"🚨 SIGNAL: NEGATIVE (All returns < {fee_bps} bps). ALLOCATE TO CASH.")
    else:
        st.success(f"✅ SIGNAL: TOP PICK IS {top_asset} (Exp. Return: {np.max(latest_pred)*100:.2f}%)")
