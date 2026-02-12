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

# --- 1. MODEL ARCHITECTURE ---
class FinancialTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, num_heads=8):
        super(FinancialTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 30, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)
        self.decoder = nn.Linear(d_model, 6) # 5 ETFs + CASH

    def forward(self, x):
        x = self.input_projection(x) + self.pos_encoder
        return self.decoder(self.transformer_encoder(x)[:, -1, :])

# --- 2. RESILIENT ENGINE ---
@st.cache_resource(ttl="1d")
def train_engine(start_year, tx_cost_bps):
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    etfs = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
    start_date = f"{start_year}-01-01"
    
    # 1. Fetch ETF Data
    try:
        raw_prices = yf.download(etfs, start=start_date, progress=False)['Close']
        if raw_prices.empty:
            st.error("Yahoo Finance data is empty. Try a different anchor year.")
            st.stop()
        returns_df = raw_prices.ffill().pct_change().dropna()
    except Exception as e:
        st.error(f"Failed to fetch ETF data: {e}")
        st.stop()
    
    # 2. Add SOFR (Cash) with Safety Defaults
    try:
        sofr_raw = fred.get_series('SOFR', start_date)
        if sofr_raw is None or sofr_raw.empty:
            returns_df['CASH'] = 0.045 / 252 # Fallback to 4.5% annual
        else:
            sofr_daily = (sofr_raw / 360 / 100).reindex(returns_df.index).ffill().fillna(0.045/252)
            returns_df['CASH'] = sofr_daily
    except:
        returns_df['CASH'] = 0.045 / 252

    # 3. Macro Features with Defensive Merging
    try:
        macro = pd.DataFrame(index=returns_df.index)
        t10y2y = fred.get_series('T10Y2Y', start_date).reindex(returns_df.index).ffill()
        macro['10Y2Y'] = t10y2y.fillna(0) # Prevent row deletion
        
        m_raw = yf.download(["^VIX", "HG=F"], start=start_date, progress=False)['Close']
        m_rets = m_raw.reindex(returns_df.index).ffill().fillna(method='bfill')
        
        full_df = pd.concat([returns_df, macro, m_rets], axis=1).ffill().dropna()
        
        if full_df.empty:
            st.error("Data Alignment Error: Resulting dataset is empty.")
            st.stop()
    except:
        st.error("Macro data merge failed.")
        st.stop()

    # 4. Multi-Day Target Creation
    target_df = returns_df.rolling(window=3).sum().shift(-3).dropna()
    common_idx = full_df.index.intersection(target_df.index)
    
    if len(common_idx) < 100:
        st.error("Insufficient overlapping data for training.")
        st.stop()
        
    full_df, target_df = full_df.loc[common_idx], target_df.loc[common_idx]

    # 5. Scaling and Training
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(full_df)
    
    def create_sequences(data, target_data, window=30):
        xs, ys = [], []
        for i in range(len(data) - window):
            xs.append(data[i:i+window])
            ys.append(target_data[i+window])
        return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))

    split_idx = int(len(scaled_data) * 0.8)
    X_train, y_train = create_sequences(scaled_data[:split_idx], target_df.values[:split_idx])
    
    model = FinancialTransformer(input_dim=full_df.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)
    loss_history = []
    
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X_train)
        loss = nn.HuberLoss()(output, y_train)
        loss.backward(); optimizer.step()
        loss_history.append(loss.item())
        
    return {"model": model, "scaler": scaler, "returns": returns_df, "features": scaled_data, "loss": loss_history}

# --- 3. UI & DASHBOARD ---
st.set_page_config(page_title="Transformer Alpha V3", layout="wide")

with st.sidebar:
    st.header("⚙️ Strategy Settings")
    regime_year = st.slider("Data Anchor", 2008, 2023, 2015)
    tx_cost = st.number_input("Tx Cost (BPS)", 0, 50, 5)
    
    engine = train_engine(regime_year, tx_cost)
    
    st.divider()
    st.subheader("Training Convergence")
    fig = go.Figure(go.Scatter(y=engine["loss"], mode='lines', line=dict(color='#00d4ff')))
    fig.update_layout(height=180, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Lower loss indicates better regime recognition.")

# 4. WALK-FORWARD INFERENCE
engine["model"].eval()
assets = ["TLT", "TBT", "VNQ", "SLV", "GLD", "CASH"]
oos_idx = int(len(engine["returns"]) * 0.8)
oos_features = engine["features"][oos_idx:]
actual_rets = engine["returns"].iloc[oos_idx:]

picks, current_pick = [], None
cost_decimal = tx_cost / 10000

for i in range(len(oos_features) - 30):
    seq = torch.FloatTensor(oos_features[i:i+30]).unsqueeze(0)
    pred = engine["model"](seq).detach().numpy()[0]
    
    # Loyalty Bonus: Current pick gets a 0.25% 'boost' to avoid unnecessary churning
    if current_pick:
        pred[assets.index(current_pick)] += (cost_decimal * 1.5) 
        
    current_pick = assets[np.argmax(pred)]
    picks.append(current_pick)

# 5. PERFORMANCE METRICS
final_rets = []
for i, p in enumerate(picks):
    day_ret = actual_rets[p].iloc[i+30]
    # Charge fee only on switch
    if i > 0 and picks[i] != picks[i-1]:
        day_ret -= cost_decimal
    final_rets.append(day_ret)

final_rets = pd.Series(final_rets, index=actual_rets.index[30:])
wealth = (1 + final_rets).cumprod()
ann_ret = ((1 + final_rets.mean())**252 - 1) * 100
sharpe = (final_rets.mean() / final_rets.std()) * np.sqrt(252)
hit_ratio = (final_rets > 0).sum() / len(final_rets) * 100

# Main Display
st.title("🚀 Transformer Alpha: Multi-Asset Dashboard")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Pick", picks[-1])
m2.metric("Ann. Return", f"{ann_ret:.1f}%")
m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
m4.metric("Hit Ratio", f"{hit_ratio:.1f}%")

st.subheader("Out-of-Sample Performance")
st.line_chart(wealth)

with st.expander("15-Day Strategy Audit"):
    audit = pd.DataFrame({
        "Date": final_rets.tail(15).index.strftime('%Y-%m-%d'),
        "Ticker": picks[-15:],
        "Net Return": [f"{v:.2%}" for v in final_rets.tail(15).values]
    })
    st.table(audit)
