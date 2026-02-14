"""
P2-TRANSFORMER-ETF-PREDICTOR (Macro-Regime Edition)
Features: VIX, TNX, DXY, Gold/Copper Ratio, GC_HG_Trend + CPI, UNRATE
Author: P2SAMAPA
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional, 
    Conv1D, GlobalMaxPooling1D, Concatenate, MultiHeadAttention, LayerNormalization
)
from sklearn.preprocessing import MinMaxScaler
from datasets import load_dataset
import plotly.graph_objects as go

# ------------------------------
# 1. UI & CONFIGURATION
# ------------------------------
st.set_page_config(page_title="P2-Transformer Alpha", layout="wide")
st.title("🚀 P2-TRANSFORMER-ETF-PREDICTOR")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Strategy Settings")
    start_year = st.slider("Training Start", 2008, 2024, 2016)
    fee_bps = st.slider("Transaction Cost (bps)", 0, 100, 15) # RESTORED
    fee_pct = fee_bps / 10000
    lookback = st.slider("Lookback Window (Days)", 10, 60, 30)
    epochs = st.number_input("Training Epochs", 10, 100, 40)
    run_button = st.button("🔥 Execute Macro-Regime Training", type="primary")

# ------------------------------
# 2. DATA PIPELINE (Enriching the 5 Signals)
# ------------------------------
@st.cache_data(ttl=3600)
def load_and_enrich_data(repo_id):
    # Load Existing (Returns, Vol, MA20, CPI, UNRATE)
    ds = load_dataset(repo_id, split="train").to_pandas()
    ds['Date'] = pd.to_datetime(ds['Date'])
    ds = ds.set_index('Date').sort_index()
    
    # Fetch 5 New Macro Signals
    st.info("🛰️ Pulling Macro Gauges (VIX, TNX, DXY, Gold, Copper)...")
    tickers = {"^VIX": "VIX", "^TNX": "TNX", "DX-Y.NYB": "DXY", "GC=F": "GOLD", "HG=F": "COPPER"}
    macro = yf.download(list(tickers.keys()), start=ds.index.min())['Close']
    macro = macro.rename(columns=tickers)
    
    # Calculate Gold/Copper Ratio & Trend
    macro['AU_CU_Ratio'] = macro['GOLD'] / macro['COPPER']
    macro['AU_CU_Trend'] = macro['AU_CU_Ratio'].rolling(window=20).mean()
    
    # Join everything
    full_df = ds.join(macro[['VIX', 'TNX', 'DXY', 'AU_CU_Ratio', 'AU_CU_Trend']], how='left')
    return full_df.ffill().dropna()

# ------------------------------
# 3. TRANSFORMER ARCHITECTURE
# ------------------------------
def build_transformer(input_shape, num_assets=5):
    inputs = Input(shape=input_shape)
    
    # Multi-Head Attention Block
    # This allows the model to "attend" to VIX vs. Price simultaneously
    attn_out = MultiHeadAttention(num_heads=4, key_dim=input_shape[1])(inputs, inputs)
    attn_out = LayerNormalization(epsilon=1e-6)(attn_out + inputs)
    
    # Temporal Convolutional Layers
    conv = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(attn_out)
    conv = GlobalMaxPooling1D()(conv)
    
    # Sequence Processing
    lstm = Bidirectional(LSTM(64, return_sequences=False))(inputs)
    
    # Feature Fusion
    merged = Concatenate()([conv, lstm])
    x = Dense(128, activation='relu')(merged)
    x = Dropout(0.2)(x)
    outputs = Dense(num_assets) # Predicting Returns for TLT, TBT, VNQ, SLV, GLD
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# ------------------------------
# 4. MAIN EXECUTION (Backtest + Training)
# ------------------------------
if run_button:
    # 1. Prepare Data
    df = load_and_enrich_data("P2SAMAPA/my-etf-data")
    target_etfs = ['TLT_Ret', 'TBT_Ret', 'VNQ_Ret', 'SLV_Ret', 'GLD_Ret']
    
    st.success(f"📈 Macro-Regime Dataset Synchronized. Rows: {len(df)}")
    
    # 2. Scale & Windowing
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(df[target_etfs].iloc[i].values)
    
    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 3. Model Training (with anti-hang UI update)
    with st.spinner("🧠 Training Transformer Attention Heads..."):
        model = build_transformer((X.shape[1], X.shape[2]))
        # Small trick: Use a simple print or standard fit to avoid UI lock
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    
    # 4. Backtesting & Prediction
    st.subheader("📊 Model Results & Predictions")
    preds = model.predict(X_test)
    
    # Logic: Pick highest predicted return. If all < 0, pick CASH.
    strategy_returns = []
    for i in range(len(preds)):
        best_asset_idx = np.argmax(preds[i])
        best_return = preds[i][best_asset_idx]
        
        if best_return > fee_pct: # Must beat transaction cost
            strategy_returns.append(y_test[i][best_asset_idx] - fee_pct)
        else:
            strategy_returns.append(0.0) # Stay in CASH (0% return)

    # 5. Visualizing Alpha
    cum_strat = np.cumprod(1 + np.array(strategy_returns))
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=cum_strat, name="Transformer Alpha Strategy"))
    st.plotly_chart(fig, use_container_width=True)

    st.write("🔥 **Current Deployment Recommendation:**")
    latest_pred = preds[-1]
    top_pick = target_etfs[np.argmax(latest_pred)]
    if np.max(latest_pred) <= fee_pct:
        st.error("🚨 ALL SIGNALS NEGATIVE: ALLOCATE TO CASH")
    else:
        st.success(f"✅ TOP SIGNAL: {top_pick.split('_')[0]} (Pred: {np.max(latest_pred)*100:.2f}%)")
