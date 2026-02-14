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
        ds = load_dataset(REPO_ID, split="train").to_pandas()
        ds['Date'] = pd.to_datetime(ds['Date'])
        ds = ds.set_index('Date').sort_index()
        
        # 5 New Signals
        tickers = {"^VIX": "VIX", "^TNX": "TNX", "DX-Y.NYB": "DXY", "GC=F": "GOLD", "HG=F": "COPPER"}
        macro = yf.download(list(tickers.keys()), start=ds.index.min())['Close']
        macro = macro.rename(columns=tickers)
        
        # Gold/Copper Ratio
        macro['AU_CU_Ratio'] = macro['GOLD'] / macro['COPPER']
        macro['AU_CU_Trend'] = macro['AU_CU_Ratio'].rolling(window=20).mean()
        
        # Join
        full_df = ds.join(macro[['VIX', 'TNX', 'DXY', 'AU_CU_Ratio', 'AU_CU_Trend']], how='left').ffill().dropna()
        
        if update_hf and HF_TOKEN:
            login(token=HF_TOKEN)
            Dataset.from_pandas(full_df.reset_index()).push_to_hub(REPO_ID)
            st.success("✅ Dataset Synced with 5 New Signals!")
        
        status.update(label="Data Ready!", state="complete")
    return full_df

# ------------------------------
# 3. TRANSFORMER MODEL ARCHITECTURE
# ------------------------------
def build_transformer(input_shape):
    # 'inputs' is our starting KerasTensor
    inputs = Input(shape=input_shape)
    
    # Attention Head
    attn = MultiHeadAttention(num_heads=4, key_dim=input_shape[1])(inputs, inputs)
    attn = LayerNormalization(epsilon=1e-6)(attn + inputs)
    
    # Temporal CNN Head
    conv = Conv1D(64, kernel_size=3, activation='relu', padding='same')(attn)
    pool = GlobalMaxPooling1D()(conv)
    
    # Recurrent Head
    lstm = Bidirectional(LSTM(64))(inputs)
    
    # Fusion
    merged = Concatenate()([pool, lstm])
    
    # Dense Layers
    x = Dense(128, activation='relu')(merged)
    x = Dropout(0.2)(x)
    
    # FIXED LINE: We call the Dense layer on 'x' to create the output KerasTensor
    outputs = Dense(5)(x) 
    
    return Model(inputs=inputs, outputs=outputs)

# ------------------------------
# 4. RUNTIME LOGIC
# ------------------------------
if run_button:
    df = get_and_sync_data()
    target_etfs = ['TLT_Ret', 'TBT_Ret', 'VNQ_Ret', 'SLV_Ret', 'GLD_Ret']
    
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

    with st.spinner("🧠 Training Hybrid Transformer..."):
        # We pass the input_shape which is (lookback, number_of_features)
        model = build_transformer((X.shape[1], X.shape[2]))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    
    # Results
    preds = model.predict(X_test)
    
    strategy_returns = []
    for i in range(len(preds)):
        best_idx = np.argmax(preds[i])
        if preds[i][best_idx] > fee_pct:
            strategy_returns.append(y_test[i][best_idx] - fee_pct)
        else:
            strategy_returns.append(0.0) # CASH

    st.subheader("📊 Performance Visualization")
    cum_strat = np.cumprod(1 + np.array(strategy_returns))
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=cum_strat, name="Transformer Alpha"))
    st.plotly_chart(fig, use_container_width=True)

    latest_pred = preds[-1]
    top_asset = target_etfs[np.argmax(latest_pred)].split('_')[0]
    
    if np.max(latest_pred) <= fee_pct:
        st.error(f"🚨 SIGNAL: CASH (Hurdle: {fee_bps} bps)")
    else:
        st.success(f"✅ SIGNAL: {top_asset} ({np.max(latest_pred)*100:.2f}%)")
