import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import os
import interface 

# [get_dual_source_data and get_live_sofr remain the same as V12.3]

@st.cache_resource(ttl="1d")
def train_high_alpha_engine(etf_list, tc_pct):
    start_date = "2012-01-01"
    train_end = "2025-08-11" # As seen in your screenshot
    
    raw_close = get_dual_source_data(etf_universe, start_date)
    if raw_close is None: return None
    
    rets = raw_close.ffill().pct_change(fill_method=None).dropna()
    
    # REWARD TUNING: Prioritizing Momentum + Volatility Stability
    roll_std = rets.rolling(20).std()
    rel_vol = roll_std / (roll_std.rolling(60).mean() + 1e-9)
    rel_vol.columns = [f"{c}_vol_ratio" for c in rel_vol.columns]
    
    full_df = pd.concat([rets, rel_vol], axis=1).dropna()
    scaler = StandardScaler()
    train_data = full_df.loc[:train_end]
    target_rets = rets.loc[train_data.index]
    
    scaled_train = scaler.fit_transform(train_data)
    agent = TacticalPPO(scaled_train.shape[1], len(etf_list))
    optimizer = optim.Adam(agent.parameters(), lr=1e-4) # Faster learning for alpha
    
    p_bar = st.progress(0, text="Recalibrating for High-Alpha...")
    for epoch in range(1000):
        idx = np.random.randint(0, len(scaled_train)-1, 128)
        # Apply Transaction Cost Penalty to the Target
        batch_target = torch.FloatTensor(target_rets.values[idx])
        
        loss = nn.MSELoss()(agent(torch.FloatTensor(scaled_train[idx])), batch_target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if epoch % 100 == 0: p_bar.progress(epoch/1000)
    p_bar.empty()

    return {"agent": agent, "scaler": scaler, "returns": rets, "features": full_df, "train_info": {"start": start_date, "end": train_end}}

# --- UI EXECUTION ---
st.set_page_config(layout="wide")

# SINGLE SLIDER DEFINITION
st.sidebar.header("🕹️ Strategy Controls")
tc_pct = st.sidebar.slider("Transaction Cost / Slippage (%)", 0.0, 1.0, 0.1, 0.05)
tc_decimal = tc_pct / 100

etf_universe = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
engine = train_high_alpha_engine(etf_universe, tc_decimal)

if engine:
    # [Prediction logic same as before]
    # ...
    
    # CALCULATE COST DRAG
    # Estimated 52 trades/year * cost per trade
    tc_drag = tc_decimal * 52 
    
    interface.render_main_output(
        top_pick, 
        f"{sharpe_val:.2f}", 
        0.50, # Hit rate placeholder
        f"{ann_ret_val:.2%}", 
        "5-Day", 
        oos_wealth, 
        audit_df, 
        engine["train_info"],
        tc_drag
    )
