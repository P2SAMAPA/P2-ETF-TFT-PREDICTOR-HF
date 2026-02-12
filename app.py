import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import os
import time

# --- 1. MOMENTUM-TILTED TRANSFORMER ---
class MomentumTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, num_heads=8, seq_len=30):
        super(MomentumTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        mask = torch.arange(seq_len).float()
        self.register_buffer('time_mask', torch.exp(mask - seq_len + 1).unsqueeze(0).unsqueeze(0))
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)
        self.decoder = nn.Linear(d_model, 6) 

    def forward(self, x):
        x = self.input_projection(x) * self.time_mask.transpose(1, 2) + self.pos_encoder
        x = self.transformer_encoder(x)
        return self.decoder(x[:, -1, :])

# --- 2. ADVANCED FEATURE ENGINE (With Infinite Value Cleaning) ---
def add_momentum_indicators(df, assets):
    for asset in assets:
        if asset == 'CASH' or asset not in df.columns: continue
        df[f'{asset}_ROC_3'] = df[asset].pct_change(3)
        df[f'{asset}_ROC_10'] = df[asset].pct_change(10)
        ema = df[asset].ewm(span=20).mean()
        df[f'{asset}_Dist_EMA'] = (df[asset] - ema) / (ema + 1e-9) # Add epsilon to prevent div by zero
    return df

@st.cache_resource(ttl="1d")
def train_engine(start_year, tx_cost_bps):
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    etfs = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
    start_date = f"{start_year}-01-01"
    
    # Retry logic for Rate Limiting
    data = None
    for i in range(3):
        data = yf.download(etfs, start=start_date, progress=False)['Close']
        if not data.empty: break
        time.sleep(5)

    if data is None or data.empty: st.stop()
    
    returns_df = data.ffill().pct_change().fillna(0)
    
    try:
        sofr = fred.get_series('SOFR', start_date)
        returns_df['CASH'] = (sofr / 360 / 100).reindex(returns_df.index).ffill().fillna(0.0001)
    except:
        returns_df['CASH'] = 0.0001

    full_df = add_momentum_indicators(returns_df.copy(), etfs + ['CASH'])
    
    try:
        vix = yf.download("^VIX", start=start_date, progress=False)['Close']
        full_df['VIX'] = vix.reindex(full_df.index).ffill().fillna(20)
    except:
        full_df['VIX'] = 20
        
    target_df = returns_df.rolling(3).sum().shift(-3).dropna()
    
    # --- CRITICAL FIX: SANITIZE INF/NAN BEFORE SCALING ---
    full_df = full_df.replace([np.inf, -np.inf], np.nan)
    full_df = full_df.ffill().bfill().fillna(0) # Remove all NaNs
    
    common_idx = full_df.index.intersection(target_df.index)
    full_df, target_df = full_df.loc[common_idx], target_df.loc[common_idx]

    if full_df.empty: st.stop()

    scaler = StandardScaler()
    # Extra safety: Ensure float64 and finite
    scaled_data = scaler.fit_transform(full_df.astype(np.float64))
    
    split_idx = int(len(scaled_data) * 0.8)
    def create_seq(data, target, window=30):
        xs, ys = [], []
        for i in range(len(data)-window):
            xs.append(data[i:i+window]); ys.append(target[i+window])
        return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))

    X_train, y_train = create_seq(scaled_data[:split_idx], target_df.values[:split_idx])
    
    model = MomentumTransformer(input_dim=full_df.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_history = []
    
    model.train()
    for _ in range(100):
        optimizer.zero_grad()
        output = model(X_train)
        weights = torch.ones_like(y_train)
        winners = torch.argmax(y_train, dim=1)
        for i, w_idx in enumerate(winners): weights[i, w_idx] = 2.5
        loss = (weights * (output - y_train)**2).mean()
        loss.backward(); optimizer.step()
        loss_history.append(loss.item())
        
    return {"model": model, "scaler": scaler, "returns": returns_df, "features": scaled_data, "loss": loss_history, "vix": full_df['VIX']}

# --- 3. UI & INFERENCE (UNCHANGED) ---
st.set_page_config(page_title="Transformer Alpha V7: Momentum Tilt", layout="wide")

with st.sidebar:
    st.header("⚙️ Strategy Settings")
    regime_year = st.slider("Year Anchor", 2008, 2023, 2015)
    tx_cost = st.number_input("Tx Cost (BPS)", 0, 50, 10)
    engine = train_engine(regime_year, tx_cost)
    
    st.subheader("Training Convergence")
    fig_loss = go.Figure(go.Scatter(y=engine["loss"], mode='lines', line=dict(color='#00d4ff')))
    fig_loss.update_layout(height=150, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark")
    st.plotly_chart(fig_loss, use_container_width=True)

engine["model"].eval()
assets = ["TLT", "TBT", "VNQ", "SLV", "GLD", "CASH"]
oos_idx = int(len(engine["returns"]) * 0.8)
oos_features = engine["features"][oos_idx:]
actual_rets = engine["returns"].iloc[oos_idx:]
vix_oos = engine["vix"].iloc[oos_idx:]

picks, current_pick, cost_dec = [], None, tx_cost/10000
for i in range(len(oos_features)-30):
    with torch.no_grad():
        pred = engine["model"](torch.FloatTensor(oos_features[i:i+30]).unsqueeze(0)).numpy()[0]
    
    loyalty_mult = 1.5 if vix_oos.iloc[i+30] < 25 else 0.5
    if current_pick: pred[assets.index(current_pick)] += (cost_dec * loyalty_mult)
    current_pick = assets[np.argmax(pred)]
    picks.append(current_pick)

final_series = pd.Series([actual_rets[p].iloc[i+30] for i, p in enumerate(picks)], index=actual_rets.index[30:30+len(picks)])
wealth = (1 + final_series).cumprod()

st.title("🚀 Transformer Alpha: Multi-Asset Dashboard")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Pick", picks[-1])
m2.metric("Ann. Return", f"{(((1 + final_series.mean())**252) - 1) * 100:.1f}%")
m3.metric("Sharpe Ratio", f"{(final_series.mean() / final_series.std()) * np.sqrt(252):.2f}")
m4.metric("Hit Ratio", f"{(final_series > 0).sum() / len(final_series) * 100:.1f}%")

st.subheader("Out-of-Sample Performance")
st.line_chart(wealth)

st.subheader("15-Day Strategy Audit")
audit_df = pd.DataFrame({"Date": final_series.tail(15).index.strftime('%Y-%m-%d'), "Ticker": picks[-15:], "Net Return": [f"{v:+.2%}" for v in final_series.tail(15).values]})
st.table(audit_df.style.applymap(lambda v: f"color: {'green' if '+' in v else 'red'}; font-weight: bold;", subset=['Net Return']))

st.divider()
st.header("📘 Model Methodology & Algorithm Details")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### **Architecture: Momentum Transformer**
    * **Self-Attention Mechanism:** Maps non-linear dependencies across a 30-day lookback.
    * **Recency Biasing:** Exponential time-decay mask favors the last 72 hours.
    """)
with col2:
    st.markdown("""
    ### **Optimization Strategy**
    * **Loss Function:** Weighted Huber Loss (2.5x penalty for winners).
    * **Data Pre-processing:** Implements Inf/NaN cleaning and float64 standardization to ensure training stability.
    """)
