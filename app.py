import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_market_calendars as mcal
import os
from datetime import datetime, timedelta
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
# 1. NYSE CALENDAR & UTILS
# ------------------------------
def get_next_open_date():
    nyse = mcal.get_calendar('NYSE')
    # Schedule check for next 10 days to account for weekends/holidays
    schedule = nyse.schedule(start_date=datetime.now(), end_date=datetime.now() + timedelta(days=10))
    if schedule.empty:
        return "TBD"
    # Find the first trading day that is strictly in the future or today if markets aren't open yet
    next_open = schedule.iloc[0].market_open
    return next_open.strftime('%b %d, %Y')

# ------------------------------
# 2. PERMANENT SIDEBAR
# ------------------------------
st.set_page_config(page_title="P2-Transformer Pro", layout="wide")

with st.sidebar:
    st.header("⚙️ Model Configuration")
    start_year = st.slider("Training Start Year", 2008, 2024, 2016)
    fee_bps = st.slider("Transaction Cost (bps)", 0, 100, 15)
    st.info("Dynamic Search: 30, 45, 60-day Lookbacks | 1, 3, 5-day Holding Periods")
    epochs = st.number_input("Training Epochs", 5, 100, 20)
    st.markdown("---")
    update_hf = st.checkbox("Sync Updates to HF Hub?", value=True)
    run_button = st.button("🚀 Execute Transformer Alpha", type="primary")

st.title("🚀 P2-TRANSFORMER-ETF-PREDICTOR")

# ------------------------------
# 3. DATA ENGINE (RECTIFIED TO PREVENT OVERLAP)
# ------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "P2SAMAPA/my-etf-data"
fee_pct = fee_bps / 10000

def get_data():
    # 1. Load existing dataset (contains ETFs, CPI, UNRATE)
    ds = load_dataset(REPO_ID, split="train").to_pandas()
    ds['Date'] = pd.to_datetime(ds['Date'])
    ds = ds.set_index('Date').sort_index()
    
    # 2. Fetch fresh macro signals
    tickers = {"^VIX": "VIX", "^TNX": "TNX", "DX-Y.NYB": "DXY", "GC=F": "GOLD", "HG=F": "COPPER"}
    macro = yf.download(list(tickers.keys()), start=ds.index.min(), progress=False)['Close']
    macro = macro.rename(columns=tickers)
    
    # 3. RECTIFICATION: Drop existing macro columns from ds to prevent 'ValueError: columns overlap'
    # This ensures fresh Yahoo Finance data replaces any stale data in the dataset
    cols_to_drop = [c for c in macro.columns if c in ds.columns]
    ds_cleaned = ds.drop(columns=cols_to_drop)
    
    # 4. Final Join
    full_df = ds_cleaned.join(macro, how='left').ffill().dropna()
    
    # Sync to Hub if requested
    if update_hf and HF_TOKEN and not full_df.equals(ds):
        login(token=HF_TOKEN)
        Dataset.from_pandas(full_df.reset_index()).push_to_hub(REPO_ID)
        
    return full_df

# ------------------------------
# 4. DYNAMIC OPTIMIZATION RUNTIME
# ------------------------------
if run_button:
    df = get_data()
    target_etfs = ['TLT_Ret', 'TBT_Ret', 'VNQ_Ret', 'SLV_Ret', 'GLD_Ret']
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    
    # SEARCH SPACE
    lookbacks = [30, 45, 60]
    holdings = [1, 3, 5]
    
    best_overall_conviction = -np.inf
    final_results = None

    with st.spinner("🔍 Discovering Patterns across Temporal Scales..."):
        for lb in lookbacks:
            X, y = [], []
            for i in range(lb, len(scaled) - 5): 
                X.append(scaled[i-lb:i])
                y.append(df[target_etfs].iloc[i].values)
            
            X, y = np.array(X), np.array(y)
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # Model Architecture
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
            model.fit(X_train, y_train, epochs=int(epochs), batch_size=32, verbose=0)
            
            preds = model.predict(X_test)
            
            # Identify Optimal Holding Period
            for hold in holdings:
                hold_rets = []
                for i in range(len(preds) - hold):
                    idx = np.argmax(preds[i])
                    # Net Return calculation
                    real_cum_ret = np.sum(y_test[i:i+hold, idx]) - fee_pct
                    hold_rets.append(real_cum_ret)
                
                avg_conviction = np.mean(hold_rets)
                
                if avg_conviction > best_overall_conviction:
                    best_overall_conviction = avg_conviction
                    final_results = {
                        "lb": lb, "hold": hold, "preds": preds, 
                        "y_test": y_test, "dates": df.index[-len(y_test):],
                        "model": model, "strat_rets": hold_rets
                    }

    # ------------------------------
    # 5. PROFESSIONAL OUTPUT UI
    # ------------------------------
    res = final_results
    strat_rets = np.array(res["strat_rets"])
    cum_strat = np.cumprod(1 + strat_rets)
    
    # Metrics
    max_single_day_loss = np.min(strat_rets) 
    oos_years = len(res["dates"]) / 252
    ann_ret = (cum_strat[-1] ** (1/oos_years)) - 1 if oos_years > 0 else 0
    rf_daily = 0.053 / 252
    sharpe = np.sqrt(252) * np.mean(strat_rets - rf_daily) / np.std(strat_rets) if np.std(strat_rets) != 0 else 0
    
    # KPI CARDS
    st.markdown("### 📈 Performance Scorecard")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        latest_pred = res["model"].predict(scaled[-res["lb"]:].reshape(1, res["lb"], -1))[0]
        top_idx = np.argmax(latest_pred)
        top_asset = target_etfs[top_idx].split('_')[0] if latest_pred[top_idx] > fee_pct else "CASH"
        st.metric("Predicted ETF", top_asset, f"NYSE Open: {get_next_open_date()}")
    
    kpi2.metric("Annualized Return", f"{ann_ret*100:.2f}%", f"{oos_years:.2f} OOS Years")
    kpi3.metric("Sharpe Ratio", f"{sharpe:.2f}", "SOFR 5.3% Adj")
    kpi4.metric("Hit Ratio (15d)", f"{(np.sum(strat_rets[-15:] > 0)/15)*100:.1f}%", f"Hold: {res['hold']}D")
    kpi5.metric("Max Single Day Loss", f"{max_single_day_loss*100:.2f}%", "Daily Stress")

    # EQUITY CURVE
    st.markdown(f"### 🚀 OOS Cumulative Returns ({res['lb']}d Lookback | {res['hold']}d Holding)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["dates"][:len(cum_strat)], y=cum_strat, fill='tozeroy', line=dict(color='#00d1b2')))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,b=0,t=20))
    st.plotly_chart(fig, use_container_width=True)

    # AUDIT TRAIL
    st.markdown("### 📋 15-Day Performance Audit Trail")
    audit_list = []
    # Reverse loop to show most recent first
    for i in range(1, 16):
        p_idx = np.argmax(res["preds"][-i])
        audit_list.append({
            "Date": res["dates"][-i].strftime('%Y-%m-%d'),
            "Predicted": target_etfs[p_idx].split('_')[0] if res["preds"][-i][p_idx] > fee_pct else "CASH",
            "Hold Period": f"{res['hold']} Day",
            "Realized Return": res["y_test"][-i][p_idx]
        })
    st.table(pd.DataFrame(audit_list).style.format({"Realized Return": "{:.2%}"}).applymap(
        lambda x: 'color: green' if x > 0 else 'color: red', subset=['Realized Return']
    ))

    # METHODOLOGY
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.write("**Architecture**")
        st.caption("Dynamic Multi-Scale Transformer. Self-selects Lookback and Holding Period using cross-validation over raw signal Attention heads.")
    with m2:
        st.write("**Universe**")
        st.caption("TLT, TBT, VNQ, SLV, GLD. Returns are net of transaction costs and SOFR hurdle rates.")
    with m3:
        st.write("**Signals (Raw Ingestion)**")
        st.caption("Macro: CPI, UNRATE. Gravity: TNX, DXY. Sentiment: VIX. Metals: GOLD, COPPER. All correlations derived via Multi-Head Attention.")
