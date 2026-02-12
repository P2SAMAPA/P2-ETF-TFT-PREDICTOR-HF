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
        self.decoder = nn.Linear(d_model, 6) 

    def forward(self, x):
        x = self.input_projection(x) + self.pos_encoder
        return self.decoder(self.transformer_encoder(x)[:, -1, :])

# --- 2. DYNAMIC ENGINE ---
@st.cache_resource(ttl="1d")
def train_engine(start_year, tx_cost_bps):
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    etfs = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
    start_date = f"{start_year}-01-01"
    
    raw_prices = yf.download(etfs, start=start_date, progress=False)['Close']
    returns_df = raw_prices.ffill().pct_change().dropna()
    
    # Dynamic Cash Fallback
    try:
        sofr = fred.get_series('SOFR', start_date)
        tbill = fred.get_series('DTB6', start_date)
        returns_df['CASH'] = (sofr.combine_first(tbill) / 360 / 100).reindex(returns_df.index).ffill().fillna(0.0001)
    except:
        returns_df['CASH'] = 0.0001

    # Dynamic Macro
    macro = pd.DataFrame(index=returns_df.index)
    try:
        macro['10Y2Y'] = fred.get_series('T10Y2Y', start_date).reindex(returns_df.index).ffill().fillna(0)
    except:
        macro['10Y2Y'] = 0
        
    m_raw = yf.download(["^VIX", "HG=F"], start=start_date, progress=False)['Close']
    macro = pd.concat([macro, m_raw.reindex(returns_df.index).ffill()], axis=1).fillna(method='bfill')
    
    full_df = pd.concat([returns_df, macro], axis=1).ffill().dropna()
    target_df = returns_df.rolling(window=3).sum().shift(-3).dropna()
    
    common_idx = full_df.index.intersection(target_df.index)
    full_df, target_df = full_df.loc[common_idx], target_df.loc[common_idx]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(full_df)
    
    split_idx = int(len(scaled_data) * 0.8)
    def create_seq(data, target, window=30):
        xs, ys = [], []
        for i in range(len(data)-window):
            xs.append(data[i:i+window]); ys.append(target[i+window])
        return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))

    X_train, y_train = create_seq(scaled_data[:split_idx], target_df.values[:split_idx])
    
    model = FinancialTransformer(input_dim=full_df.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)
    loss_history = []
    
    model.train()
    for _ in range(100):
        optimizer.zero_grad()
        loss = nn.HuberLoss()(model(X_train), y_train)
        loss.backward(); optimizer.step()
        loss_history.append(loss.item())
        
    return {"model": model, "scaler": scaler, "returns": returns_df, "features": scaled_data, "loss": loss_history}

# --- 3. UI ---
st.set_page_config(page_title="Transformer Alpha V5", layout="wide")
with st.sidebar:
    regime_year = st.slider("Data Anchor", 2008, 2023, 2015)
    tx_cost = st.number_input("Tx Cost (BPS)", 0, 50, 10)
    engine = train_engine(regime_year, tx_cost)

# 4. INFERENCE
engine["model"].eval()
assets = ["TLT", "TBT", "VNQ", "SLV", "GLD", "CASH"]
oos_idx = int(len(engine["returns"]) * 0.8)
oos_features = engine["features"][oos_idx:]
actual_rets_df = engine["returns"].iloc[oos_idx:]

picks, current_pick, cost_decimal = [], None, tx_cost / 10000
for i in range(len(oos_features) - 30):
    pred = engine["model"](torch.FloatTensor(oos_features[i:i+30]).unsqueeze(0)).detach().numpy()[0]
    if current_pick: pred[assets.index(current_pick)] += (cost_decimal * 2.0)
    current_pick = assets[np.argmax(pred)]
    picks.append(current_pick)

available_rets = actual_rets_df.iloc[30:].iloc[:len(picks)]
final_rets = []
for i, p in enumerate(picks):
    day_ret = available_rets[p].iloc[i]
    if i > 0 and picks[i] != picks[i-1]: day_ret -= cost_decimal
    final_rets.append(day_ret)

final_series = pd.Series(final_rets, index=available_rets.index)
st.title("🚀 Transformer Alpha V5")
st.line_chart((1 + final_series).cumprod())

# --- 5. COLOR-CODED AUDIT TRAIL ---
st.subheader("15-Day Strategy Audit")

def color_returns(val):
    color = 'red' if '-' in val else 'green'
    return f'color: {color}; font-weight: bold;'

audit_data = pd.DataFrame({
    "Date": final_series.tail(15).index.strftime('%Y-%m-%d'),
    "Ticker": picks[-15:],
    "Net Return": [f"{v:+.2%}" for v in final_series.tail(15).values]
})

# Applying the style
st.table(audit_data.style.applymap(color_returns, subset=['Net Return']))
