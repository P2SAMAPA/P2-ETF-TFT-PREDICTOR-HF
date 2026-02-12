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

# --- 1. CONFIGURATION & MODELS ---
class FinancialTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, num_heads=8):
        super(FinancialTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 30, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)
        self.decoder = nn.Linear(d_model, 6) # TLT, TBT, VNQ, SLV, GLD + CASH

    def forward(self, x):
        x = self.input_projection(x) + self.pos_encoder
        return self.decoder(self.transformer_encoder(x)[:, -1, :])

# --- 2. THE ENGINE (Logic) ---
@st.cache_resource(ttl="1d")
def train_engine(start_year, tx_cost_bps):
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    etfs = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
    start_date = f"{start_year}-01-01"
    
    # 1. Fetch Data
    raw_prices = yf.download(etfs, start=start_date, progress=False)['Close']
    returns_df = raw_prices.ffill().pct_change().dropna()
    
    # 2. Add SOFR (Cash)
    sofr = (fred.get_series('SOFR', start_date) / 360 / 100).reindex(returns_df.index).ffill().fillna(0.0001)
    returns_df['CASH'] = sofr

    # 3. Macro Features
    macro = pd.DataFrame(index=returns_df.index)
    macro['10Y2Y'] = fred.get_series('T10Y2Y', start_date).reindex(returns_df.index).ffill()
    m_raw = yf.download(["^VIX", "HG=F"], start=start_date, progress=False)['Close']
    full_df = pd.concat([returns_df, macro, m_raw.reindex(returns_df.index).ffill()], axis=1).dropna()

    # 4. Multi-Day Target (Predicting 3-day returns to reduce churning)
    target_df = returns_df.rolling(window=3).sum().shift(-3).dropna()
    common_idx = full_df.index.intersection(target_df.index)
    full_df, target_df = full_df.loc[common_idx], target_df.loc[common_idx]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(full_df)
    
    def create_sequences(data, target_data, window=30):
        xs, ys = [], []
        for i in range(len(data) - window):
            xs.append(data[i:i+window]); ys.append(target_data[i+window])
        return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))

    split_idx = int(len(scaled_data) * 0.8)
    X_train, y_train = create_sequences(scaled_data[:split_idx], target_df.values[:split_idx])
    
    # 5. Training
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

# --- 3. THE INTERFACE (Visuals) ---
st.set_page_config(page_title="Transformer Alpha: Multi-Day Regime", layout="wide", initial_sidebar_state="expanded")

# Sidebar
with st.sidebar:
    st.title("⚙️ Parameters")
    regime_year = st.slider("Data Anchor (History)", 2008, 2023, 2015)
    tx_cost = st.number_input("Transaction Cost (BPS)", 0, 50, 5)
    st.divider()
    
    # Run/Cache Logic
    engine = train_engine(regime_year, tx_cost)
    
    st.subheader("Model Diagnostics")
    fig = go.Figure(go.Scatter(y=engine["loss"], mode='lines', line=dict(color='#00d4ff')))
    fig.update_layout(height=180, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", xaxis_title="Epoch", yaxis_title="Loss")
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 **Target:** 3-Day Lookahead. Predicts the best asset for the next 72 hours.")

# Main Dashboard Simulation
engine["model"].eval()
assets = ["TLT", "TBT", "VNQ", "SLV", "GLD", "CASH"]
oos_idx = int(len(engine["returns"]) * 0.8)
oos_features = engine["features"][oos_idx:]
actual_rets = engine["returns"].iloc[oos_idx:]

# Run Walk-Forward Inference
picks, current_pick = [], None
cost_decimal = tx_cost / 10000

for i in range(len(oos_features) - 30):
    seq = torch.FloatTensor(oos_features[i:i+30]).unsqueeze(0)
    pred = engine["model"](seq).detach().numpy()[0]
    if current_pick: pred[assets.index(current_pick)] += (cost_decimal * 0.5) # Loyalty Bonus
    current_pick = assets[np.argmax(pred)]
    picks.append(current_pick)

# Backtest Math
final_rets = []
for i, p in enumerate(picks):
    day_ret = actual_rets[p].iloc[i+30]
    if i > 0 and picks[i] != picks[i-1]: day_ret -= cost_decimal
    final_rets.append(day_ret)

final_rets = pd.Series(final_rets, index=actual_rets.index[30:])
wealth = (1 + final_rets).cumprod()

# Display Results
c1, c2, c3 = st.columns(3)
c1.metric("Top Alpha Pick", picks[-1])
c2.metric("Ann. Return", f"{((1+final_rets.mean())**252 - 1)*100:.1f}%")
c3.metric("Sharpe Ratio", round((final_rets.mean() / final_rets.std()) * np.sqrt(252), 2))

st.subheader("Strategy Cumulative Wealth")
st.line_chart(wealth)

with st.expander("Detailed 15-Day Audit Trail"):
    audit = pd.DataFrame({"Date": final_rets.tail(15).index.strftime('%Y-%m-%d'), "Ticker": picks[-15:], "Daily Net": [f"{v:.2%}" for v in final_rets.tail(15).values]})
    st.table(audit)
