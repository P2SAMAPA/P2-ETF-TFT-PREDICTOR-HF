import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from polygon import RESTClient
import yfinance as yf
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# ... [Keep your existing API Key and LSTMModel Class code here] ...

# ==========================================
# 1. UI SETUP (Clean & Professional)
# ==========================================
st.set_page_config(page_title="Fixed Income Alpha", layout="wide")
st.title("🏦 Fixed Income Alpha Optimizer")
st.markdown("Automated LSTM Prediction & Holding Period Optimization")

# Sidebar: Transaction Costs & Data
st.sidebar.header("Trading Constraints")
tx_cost_bps = st.sidebar.slider("Transaction Cost (bps)", 0, 100, 10)
lookback_days = 1825  # Hardcoded 5 years
fixed_income_etfs = ["VCLT", "TLT", "TBT", "MBB", "VNQ", "HYG", "SLV", "GLD", "PFF"]

# ==========================================
# 2. AUTOMATED ENGINE
# ==========================================
data = get_hybrid_data(fixed_income_etfs, lookback_days)

if not data.empty:
    # Auto-Train on Page Load
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.dropna())
    
    # Simple training wrapper to keep UI clean
    def train_automated(scaled_data):
        window = 30
        X, y = [], []
        for i in range(len(scaled_data) - window):
            X.append(scaled_data[i:i+window])
            y.append(scaled_data[i+window])
        
        model = LSTMModel(len(fixed_income_etfs), 64, 2, len(fixed_income_etfs))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Train for 50 epochs (balanced for speed/accuracy)
        for _ in range(50):
            model.train()
            optimizer.zero_grad()
            out = model(torch.FloatTensor(np.array(X)))
            loss = criterion(out, torch.FloatTensor(np.array(y)))
            loss.backward()
            optimizer.step()
        return model, scaler

    model, scaler = train_automated(scaled_data)

    # ==========================================
    # 3. ANALYSIS & OPTIMIZATION
    # ==========================================
    model.eval()
    last_window = torch.FloatTensor(scaled_data[-30:].reshape(1, 30, -1))
    raw_pred = model(last_window).detach().numpy()
    pred_prices = scaler.inverse_transform(raw_pred)[0]
    
    current_prices = data.iloc[-1].values
    expected_returns = (pred_prices - current_prices) / current_prices
    
    # Calculate Net Returns for 1d, 3d, 5d (Simulated drift)
    results = []
    cost_pct = tx_cost_bps / 10000
    
    for i, ticker in enumerate(fixed_income_etfs):
        ret_1d = expected_returns[i] - cost_pct
        ret_3d = (expected_returns[i] * 1.5) - cost_pct # Simplified drift
        ret_5d = (expected_returns[i] * 2.2) - cost_pct
        
        best_ret = max(ret_1d, ret_3d, ret_5d)
        best_h = "1 Day" if best_ret == ret_1d else "3 Days" if best_ret == ret_3d else "5 Days"
        
        results.append({"ETF": ticker, "Net Return": best_ret, "Horizon": best_h})

    res_df = pd.DataFrame(results).sort_values(by="Net Return", ascending=False)
    top_pick = res_df.iloc[0]

    # ==========================================
    # 4. FINAL DASHBOARD UI
    # ==========================================
    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("Top Alpha Pick", top_pick['ETF'], f"{top_pick['Net Return']:.2%}")
        st.write(f"**Recommended Holding:** {top_pick['Horizon']}")
        st.write(f"**Net of {tx_cost_bps} bps cost**")
        
        if st.button("🔄 Re-Run Optimization"):
            st.rerun()

    with col2:
        st.subheader("Opportunity Ranking")
        st.dataframe(res_df.style.format({"Net Return": "{:.2%}"}), use_container_width=True)

    # Performance Chart
    st.divider()
    fig = go.Figure()
    for t in fixed_income_etfs:
        fig.add_trace(go.Scatter(x=data.index[-100:], y=data[t].tail(100), name=t))
    fig.update_layout(title="Recent Performance (Last 100 Days)", template="plotly_white")
    st.plotly_chart(fig, width='stretch')
