import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
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
# 1. PERMANENT SIDEBAR (UNTOUCHED)
# ------------------------------
st.set_page_config(page_title="P2-Transformer Alpha", layout="wide")

with st.sidebar:
    st.header("⚙️ Model Configuration")
    start_year = st.slider("Training Start Year", 2008, 2024, 2016)
    fee_bps = st.slider("Transaction Cost (bps)", 0, 100, 15)
    lookback = st.slider("Lookback Window (Days)", 10, 60, 30)
    epochs = st.number_input("Training Epochs", 5, 100, 20)
    st.markdown("---")
    update_hf = st.checkbox("Sync Updates to HF Hub?", value=True)
    run_button = st.button("🚀 Execute Transformer Alpha", type="primary")

st.title("🚀 P2-TRANSFORMER-ETF-PREDICTOR")

# ------------------------------
# 2. DATA ENGINE (FIXED: CPI & UNRATE INCLUDED)
# ------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "P2SAMAPA/my-etf-data"
fee_pct = fee_bps / 10000

def get_and_sync_data():
    # Load dataset containing core ETF returns + CPI/UNRATE
    ds = load_dataset(REPO_ID, split="train").to_pandas()
    ds['Date'] = pd.to_datetime(ds['Date'])
    ds = ds.set_index('Date').sort_index()
    
    # Live Macro Tickers for Regime Filtering
    tickers = {"^VIX": "VIX", "^TNX": "TNX", "DX-Y.NYB": "DXY", "GC=F": "GOLD", "HG=F": "COPPER"}
    try:
        macro = yf.download(list(tickers.keys()), start=ds.index.min(), progress=False)['Close']
        macro = macro.rename(columns=tickers)
        if 'GOLD' in macro.columns and 'COPPER' in macro.columns:
            macro['AU_CU_Ratio'] = macro['GOLD'] / macro['COPPER']
            macro['AU_CU_Trend'] = macro['AU_CU_Ratio'].rolling(window=20).mean()
        
        # Merge macro signals into existing dataset (which includes CPI, UNRATE)
        macro_cols = ['VIX', 'TNX', 'DXY', 'AU_CU_Ratio', 'AU_CU_Trend']
        cols_to_add = [c for c in macro.columns if c in macro_cols and c not in ds.columns]
        full_df = ds.join(macro[cols_to_add], how='left') if cols_to_add else ds
        full_df = full_df.ffill().dropna()
        
        if update_hf and HF_TOKEN and not full_df.equals(ds):
            login(token=HF_TOKEN)
            Dataset.from_pandas(full_df.reset_index()).push_to_hub(REPO_ID)
        return full_df
    except:
        return ds

# ------------------------------
# 3. CORE RUNTIME (MODELING & PERFORMANCE)
# ------------------------------
if run_button:
    df = get_and_sync_data()
    target_etfs = ['TLT_Ret', 'TBT_Ret', 'VNQ_Ret', 'SLV_Ret', 'GLD_Ret']
    
    # Scaler & Tensors
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(df[target_etfs].iloc[i].values)
    X, y = np.array(X), np.array(y)

    # Split 80:20 for proper OOS segmentation
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    oos_dates = df.index[-len(X_test):]

    # Model
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
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    
    # Inference on OOS
    preds = model.predict(X_test)
    
    # Performance Calculations
    strat_rets = []
    audit_data = []
    for i in range(len(preds)):
        idx = np.argmax(preds[i])
        real_ret = y_test[i][idx]
        daily_pnl = (real_ret - fee_pct) if preds[i][idx] > fee_pct else 0.0
        strat_rets.append(daily_pnl)
        
        if i >= len(preds) - 15:
            audit_data.append({
                "Date": oos_dates[i].strftime('%Y-%m-%d'),
                "Predicted": target_etfs[idx].split('_')[0] if preds[i][idx] > fee_pct else "CASH",
                "Realized Return": real_ret
            })

    strat_rets = np.array(strat_rets)
    cum_strat = np.cumprod(1 + strat_rets)
    
    # 1. OOS Year Count
    oos_years = len(oos_dates) / 252
    
    # 2. Annualized Return
    ann_return = (cum_strat[-1] ** (1 / oos_years)) - 1 if oos_years > 0 else 0
    
    # 3. Correct Max Drawdown
    peak = np.maximum.accumulate(cum_strat)
    drawdown = (cum_strat - peak) / peak
    max_dd = np.min(drawdown) # Standard Peak-to-Trough calculation

    # 4. Sharpe (SOFR 5.3%)
    rf_daily = 0.053 / 252
    sharpe = np.sqrt(252) * np.mean(strat_rets - rf_daily) / np.std(strat_rets) if np.std(strat_rets) != 0 else 0
    
    # 5. Hit Ratio (Last 15 days)
    recent_rets = strat_rets[-15:]
    hit_ratio = (np.sum(recent_rets > 0) / 15) * 100

    # ------------------------------
    # 4. PROFESSIONAL OUTPUT UI (FIXED)
    # ------------------------------
    st.markdown("### 📈 Performance Scorecard")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    # Prediction Header
    top_idx = np.argmax(model.predict(X[-1:])[0])
    final_pred = target_etfs[top_idx].split('_')[0] if model.predict(X[-1:])[0][top_idx] > fee_pct else "CASH"
    kpi1.metric("Predicted ETF", final_pred, f"US Open {datetime.now().strftime('%b %d, %Y')}")
    
    kpi2.metric("Annualized Return", f"{ann_return*100:.2f}%", f"{oos_years:.2f} OOS Years")
    kpi3.metric("Sharpe Ratio", f"{sharpe:.2f}", "SOFR Adjusted")
    kpi4.metric("Hit Ratio (15d)", f"{hit_ratio:.1f}%", f"{np.sum(recent_rets > 0)}/15 Days")
    kpi5.metric("Max Drawdown", f"{max_dd*100:.2f}%", "Daily Peak-to-Trough")

    # --- OOS GRAPH ---
    st.markdown("### 🚀 Out-of-Sample (OOS) Cumulative Returns")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=oos_dates, y=cum_strat, fill='tozeroy', line=dict(color='#00d1b2', width=2)))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,b=0,t=20))
    st.plotly_chart(fig, use_container_width=True)

    # --- AUDIT TRAIL ---
    st.markdown("### 📋 15-Day Performance Audit Trail")
    audit_df = pd.DataFrame(audit_data)
    st.table(audit_df.style.format({"Realized Return": "{:.2%}"}).applymap(
        lambda x: 'color: green' if x > 0 else 'color: red', subset=['Realized Return']
    ))

    # --- METHODOLOGY (UPDATED SIGNALS) ---
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.write("**Architecture**")
        st.caption("Hybrid Transformer-LSTM with Multi-Head Attention. Optimizes for cross-asset correlations and macro regimes.")
    with m2:
        st.write("**ETFs Used**")
        st.caption("TLT, TBT, VNQ, SLV, GLD.")
    with m3:
        st.write("**Signals (Gravity & Regime)**")
        st.caption("Macro: CPI (Inflation), UNRATE (Labor). Sentiment: VIX, DXY. Gravity: TNX (10Y Yield). Barometer: Gold/Copper Ratio.")
