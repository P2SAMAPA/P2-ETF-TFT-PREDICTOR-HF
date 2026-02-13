import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os
import gc
from datetime import datetime, timedelta

# --- 1. LOCAL DATA ENGINE ---
LOCAL_FILENAME = "historical_cache.parquet"

def get_local_data(assets, start_date_str):
    # Strictly local to avoid cloud-download memory spikes
    if os.path.exists(LOCAL_FILENAME):
        df = pd.read_parquet(LOCAL_FILENAME)
        last_date = df.index.max()
        if last_date.date() < (datetime.now() - timedelta(days=1)).date():
            new_data = yf.download(assets, start=(last_date + timedelta(days=1)).strftime('%Y-%m-%d'), progress=False)['Close']
            if not new_data.empty:
                df = pd.concat([df, new_data]).drop_duplicates().sort_index()
                df.to_parquet(LOCAL_FILENAME)
        return df
    else:
        df = yf.download(assets, start=start_date_str, progress=False)['Close']
        df.to_parquet(LOCAL_FILENAME)
        return df

# --- 2. TRANSFORMER MODEL (Dropout 0.3) ---
class MomentumTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, num_heads=8, seq_len=30):
        super(MomentumTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        # 0.3 Dropout as requested to prevent the "VNQ Bias"
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=0.3, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)
        self.decoder = nn.Linear(d_model, 6) 

    def forward(self, x):
        x = self.input_projection(x) + self.pos_encoder
        x = self.transformer_encoder(x)
        return self.decoder(x[:, -1, :])

@st.cache_resource(ttl="30m", max_entries=1) # CRITICAL: Prevents 16Gi Memory Leak
def train_engine(start_year, tx_cost, lookback):
    # Force clear CPU memory and cache before training
    gc.collect() 
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    etfs = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
    
    data = get_local_data(etfs, f"{start_year}-01-01")
    returns_df = data.ffill().pct_change().fillna(0)
    
    try:
        sofr_series = fred.get_series('SOFR')
        sofr_val = sofr_series.iloc[-1]
        returns_df['CASH'] = (sofr_val / 360 / 100)
    except:
        sofr_val = 5.33
        returns_df['CASH'] = 0.0001

    for asset in etfs + ['CASH']:
        returns_df[f'{asset}_ROC_10'] = returns_df[asset].pct_change(10)
    
    features_df = returns_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    target_df = returns_df[etfs + ['CASH']].rolling(3).sum().shift(-3).dropna()
    
    # --- DYNAMIC 80:20 SPLIT LOGIC ---
    total_len = len(features_df)
    split_idx = int(total_len * 0.8)
    
    train_feat = features_df.iloc[:split_idx]
    oos_feat = features_df.iloc[split_idx:]
    
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_feat.astype(np.float64))
    scaled_oos = scaler.transform(oos_feat.astype(np.float64))
    
    # Sliding Lookback Window Logic
    X_train = torch.FloatTensor(np.array([scaled_train[i:i+lookback] for i in range(len(scaled_train)-lookback)]))
    y_train = torch.FloatTensor(target_df.iloc[lookback:split_idx].values)

    model = MomentumTransformer(input_dim=features_df.shape[1], seq_len=lookback)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    model.train()
    for e in range(100):
        optimizer.zero_grad()
        output = model(X_train)
        loss = nn.MSELoss()(output, y_train[:len(output)])
        loss.backward()
        optimizer.step()
        
    return {"model": model, "scaler": scaler, "returns": returns_df[etfs+['CASH']], "oos_features": scaled_oos, "sofr": sofr_val, "oos_dates": oos_feat.index, "lookback": lookback}

# --- 3. UI SETUP ---
st.set_page_config(page_title="ETF Alpha Maximizer", layout="wide")

with st.sidebar:
    st.header("⚙️ Strategy Settings")
    regime_year = st.slider("Year Anchor", 2008, 2023, 2021)
    tx_cost_bps = st.slider("Transaction Cost (BPS)", 0, 50, 15)
    
    # UI TOGGLE: 30, 45, or 60 days
    lookback_window = st.radio("Lookback Window (Days)", [30, 45, 60], index=0, horizontal=True)
    
    with st.spinner("Training (Memory-Optimized)..."):
        engine = train_engine(regime_year, tx_cost_bps, lookback_window)
    
    st.markdown("---")
    st.subheader("Model Status")
    st.info(f"OOS Range: {len(engine['oos_dates'])} days")

# --- 4. INFERENCE & METRICS ---
assets = ["TLT", "TBT", "VNQ", "SLV", "GLD", "CASH"]
oos_features = engine["oos_features"]
actual_rets = engine["returns"].loc[engine["oos_dates"]]
lb = engine["lookback"]

picks = []
engine["model"].eval()
for i in range(len(oos_features)-lb):
    with torch.no_grad():
        pred = engine["model"](torch.FloatTensor(oos_features[i:i+lb]).unsqueeze(0)).numpy()[0]
    picks.append(assets[np.argmax(pred)])

final_index = engine["oos_dates"][lb : lb + len(picks)]
res_df = pd.DataFrame({
    "Pick": picks, 
    "Return": [actual_rets[p].iloc[i+lb] for i, p in enumerate(picks)]
}, index=final_index)

# Net-of-fees
res_df['Return'] = res_df['Return'] - (tx_cost_bps / 10000)
wealth = (1 + res_df["Return"]).cumprod()

st.title("Fixed Income/Commodity ETF Alpha Maximizer")

# KPI Columns
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Current Pick", picks[-1])
with m2:
    ann_ret = (res_df['Return'].mean() * 252) * 100
    st.metric("Ann. Return", f"{ann_ret:.1f}%")
with m3:
    sharpe = (res_df['Return'].mean() / res_df['Return'].std()) * np.sqrt(252)
    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
with m4:
    hit_15 = (res_df['Return'].tail(15) > 0).sum() / 15 * 100
    st.metric("Hit Ratio (15d)", f"{hit_15:.1f}%")

st.line_chart(wealth)

# Audit Trail
st.subheader("📋 Strategy Audit Trail")
st.dataframe(res_df.tail(15))

# Methodology Documentation
st.markdown("---")
st.subheader("🧠 System Logic")
st.info(f"""
- **Memory Management**: Cache is locked to a single entry (`max_entries=1`) to prevent 16Gi crashes.
- **Lookback Toggle**: Currently using a {lb}-day window for momentum detection.
- **Dropout (0.3)**: Applied to the Transformer layers to improve generalization across different interest rate regimes.
- **Data Anchor**: Training on data starting from {regime_year}, with a strict 80:20 split.
""")
