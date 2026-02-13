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
from huggingface_hub import hf_hub_download, HfApi

# --- 1. REPRODUCIBILITY & SEEDING ---
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Ensure consistent picks across sessions
    torch.use_deterministic_algorithms(True, warn_only=True)

set_seed(42)

# --- 2. DATA PERSISTENCE ENGINE ---
REPO_ID = "P2SAMAPA/etf-alpha-data" 
FILENAME = "historical_cache.parquet"
HF_TOKEN = os.getenv("HF_TOKEN")

def sync_data_persistent(assets, start_date_str):
    try:
        path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset", token=HF_TOKEN)
        df = pd.read_parquet(path)
        last_date = df.index.max()
        # Automatic daily update check
        if last_date.date() < (datetime.now() - timedelta(days=1)).date():
            new_data = yf.download(assets, start=(last_date + timedelta(days=1)).strftime('%Y-%m-%d'), progress=False)['Close']
            if not new_data.empty:
                df = pd.concat([df, new_data]).drop_duplicates().sort_index()
                df.to_parquet(FILENAME)
                HfApi().upload_file(path_or_fileobj=FILENAME, path_in_repo=FILENAME, repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
        return df
    except:
        return yf.download(assets, start=start_date_str, progress=False)['Close']

# --- 3. HIGH-CAPACITY TRANSFORMER MODEL ---
class MomentumTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, num_heads=8, seq_len=30):
        super(MomentumTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        # Deep 3-layer architecture for pattern detection
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)
        self.decoder = nn.Linear(d_model, 6) 

    def forward(self, x):
        x = self.input_projection(x) + self.pos_encoder
        x = self.transformer_encoder(x)
        return self.decoder(x[:, -1, :])

@st.cache_resource(ttl="1h")
def train_engine(start_year, tx_cost):
    gc.collect()
    set_seed(42)
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    etfs = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
    
    # Pool starts at Anchor Year
    data = sync_data_persistent(etfs, f"{start_year}-01-01")
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
    
    # --- DYNAMIC 80:20 MATH ---
    # Calculates exact split based on the length of the data pool
    total_len = len(features_df)
    split_idx = int(total_len * 0.8)
    
    train_feat = features_df.iloc[:split_idx]
    oos_feat = features_df.iloc[split_idx:]
    
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_feat.astype(np.float64))
    scaled_oos = scaler.transform(oos_feat.astype(np.float64))
    
    # Windowing for training
    X_train = torch.FloatTensor(np.array([scaled_train[i:i+30] for i in range(len(scaled_train)-30)]))
    y_train = torch.FloatTensor(target_df.iloc[30:split_idx].values)

    model = MomentumTransformer(input_dim=features_df.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    model.train()
    for e in range(100):
        optimizer.zero_grad()
        output = model(X_train)
        loss = nn.MSELoss()(output, y_train[:len(output)])
        loss.backward(); optimizer.step()
        
    return {"model": model, "scaler": scaler, "returns": returns_df[etfs+['CASH']], "oos_features": scaled_oos, "sofr": sofr_val, "oos_dates": oos_feat.index}

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="ETF Alpha Maximizer", layout="wide")

with st.sidebar:
    st.header("⚙️ Strategy Settings")
    regime_year = st.slider("Year Anchor (Pool Start)", 2008, 2023, 2021)
    tx_cost_bps = st.slider("Transaction Cost (BPS)", 0, 50, 15)
    
    with st.spinner("Training Deep Transformer..."):
        engine = train_engine(regime_year, tx_cost_bps)
    
    st.markdown("---")
    st.subheader("Model Status")
    st.success("Seeded: 42 (Reproducible)")
    st.info(f"OOS Range: {len(engine['oos_dates'])} days")

# --- 5. INFERENCE & ALIGNMENT ---
assets = ["TLT", "TBT", "VNQ", "SLV", "GLD", "CASH"]
oos_features = engine["oos_features"]
actual_rets = engine["returns"].loc[engine["oos_dates"]]

picks = []
engine["model"].eval()
for i in range(len(oos_features)-30):
    with torch.no_grad():
        pred = engine["model"](torch.FloatTensor(oos_features[i:i+30]).unsqueeze(0)).numpy()[0]
    picks.append(assets[np.argmax(pred)])

# Alignment logic for time-series consistency
final_index = engine["oos_dates"][30 : 30 + len(picks)]
res_df = pd.DataFrame({
    "Pick": picks, 
    "Return": [actual_rets[p].iloc[i+30] for i, p in enumerate(picks)]
}, index=final_index)

# Net-of-Fees Calculation
res_df['Return'] = res_df['Return'] - (tx_cost_bps / 10000)
wealth = (1 + res_df["Return"]).cumprod()

st.title("Fixed Income/Commodity ETF Alpha Maximizer")

# Metrics Display
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"**Current Pick**\n<h2 style='margin:0;'>{picks[-1]}</h2><p style='font-size:12px; color:blue;'>Hold Type: 3-Day Target</p>", unsafe_allow_html=True)
with m2:
    ann_ret = (res_df['Return'].mean() * 252) * 100
    oos_yrs = round(len(res_df)/252, 1)
    st.markdown(f"**Ann. Return**\n<h2 style='margin:0;'>{ann_ret:.1f}%</h2><p style='font-size:12px; color:gray;'>{oos_yrs}y OOS (Exact 20%)</p>", unsafe_allow_html=True)
with m3:
    sharpe = (res_df['Return'].mean() / res_df['Return'].std()) * np.sqrt(252)
    st.markdown(f"**Sharpe Ratio**\n<h2 style='margin:0;'>{sharpe:.2f}</h2><p style='font-size:12px; color:green;'>SOFR: {engine['sofr']}%</p>", unsafe_allow_html=True)
with m4:
    hit_15 = (res_df['Return'].tail(15) > 0).sum() / 15 * 100
    st.markdown(f"**Hit Ratio**\n<h2 style='margin:0;'>{hit_15:.1f}%</h2><p style='font-size:12px; color:gray;'>Last 15 Trading Days</p>", unsafe_allow_html=True)

st.line_chart(wealth)

# Audit Trail Table
st.subheader("📋 15-Day Strategy Audit Trail")
audit_data = res_df.tail(15).copy()
audit_data['Return'] = audit_data['Return'].apply(lambda x: f"{x*100:+.2f}%")
def style_returns(val):
    return 'color: green; font-weight: bold;' if '+' in val else 'color: red; font-weight: bold;'
st.table(audit_data.style.applymap(style_returns, subset=['Return']))

# Methodology Section
st.markdown("---")
st.subheader("🧠 Methodology & Process")
st.info(f"""
- **Seeded Execution**: Hard-coded Seed 42 ensures the Transformer weights initialize identically for reproducible picks.
- **Dynamic 80:20 Slicing**: The training/testing boundary is calculated as a fraction of the total days from {regime_year} to present.
- **Alpha Engine**: Reinstated the original high-capacity 3-layer Transformer architecture.
- **Data Fidelity**: Historical data is synced daily from live feeds; transaction costs are applied daily as net-of-fee slippage.
""")
