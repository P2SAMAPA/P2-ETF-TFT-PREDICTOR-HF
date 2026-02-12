import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os

# --- 1. GROWTH-FOCUSED TRANSFORMER ---
class AbsoluteTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, num_heads=8):
        super(AbsoluteTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 30, d_model))
        # Lower dropout to allow more "aggressive" learning from recent peaks
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)
        self.decoder = nn.Linear(d_model, 6) 

    def forward(self, x):
        x = self.input_projection(x) + self.pos_encoder
        return self.decoder(self.transformer_encoder(x)[:, -1, :])

# --- 2. ENGINE (MAX RETURN OPTIMIZATION) ---
@st.cache_resource(ttl="1d")
def train_engine(start_year, tx_cost_bps):
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    etfs = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
    start_date = f"{start_year}-01-01"
    
    raw_prices = yf.download(etfs, start=start_date, progress=False)['Close']
    returns_df = raw_prices.ffill().pct_change().dropna()
    
    # Tiered Cash Fallback
    try:
        sofr = fred.get_series('SOFR', start_date)
        tbill = fred.get_series('DTB6', start_date)
        returns_df['CASH'] = (sofr.combine_first(tbill) / 360 / 100).reindex(returns_df.index).ffill().fillna(0.0001)
    except:
        returns_df['CASH'] = 0.0001

    # Macro Input
    macro = pd.DataFrame(index=returns_df.index)
    macro['10Y2Y'] = fred.get_series('T10Y2Y', start_date).reindex(returns_df.index).ffill().fillna(0)
    vix = yf.download("^VIX", start=start_date, progress=False)['Close']
    macro['VIX'] = vix.reindex(returns_df.index).ffill().fillna(20)
    
    full_df = pd.concat([returns_df, macro], axis=1).ffill().dropna()
    
    # TARGET: Raw 3-Day Future Return (Prioritizes Velocity over Safety)
    target_df = returns_df.rolling(3).sum().shift(-3).dropna()
    
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
    
    model = AbsoluteTransformer(input_dim=full_df.shape[1])
    # Higher Learning Rate for faster adaptation to breakout trends
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    
    model.train()
    for _ in range(100):
        optimizer.zero_grad()
        loss = nn.HuberLoss()(model(X_train), y_train)
        loss.backward(); optimizer.step()
        
    return {"model": model, "scaler": scaler, "returns": returns_df, "features": scaled_data}

# --- 3. UI & INFERENCE ---
st.set_page_config(page_title="Transformer Alpha V7: Max Return", layout="wide")
with st.sidebar:
    regime_year = st.slider("Year", 2008, 2023, 2015)
    tx_cost = st.number_input("BPS Cost", 0, 50, 10)
    engine = train_engine(regime_year, tx_cost)

# Inference Logic
engine["model"].eval()
assets = ["TLT", "TBT", "VNQ", "SLV", "GLD", "CASH"]
oos_idx = int(len(engine["returns"]) * 0.8)
oos_features = engine["features"][oos_idx:]
actual_rets = engine["returns"].iloc[oos_idx:]

picks, current_pick, cost_decimal = [], None, tx_cost/10000
for i in range(len(oos_features)-30):
    with torch.no_grad():
        pred = engine["model"](torch.FloatTensor(oos_features[i:i+30]).unsqueeze(0)).numpy()[0]
    
    # Reduced Loyalty Bonus to allow faster rotation into "hot" assets
    if current_pick: pred[assets.index(current_pick)] += (cost_decimal * 1.5)
    current_pick = assets[np.argmax(pred)]
    picks.append(current_pick)

# Match the index length to the picks length exactly
final_series = pd.Series(
    [actual_rets[p].iloc[i+30] for i, p in enumerate(picks)], 
    index=actual_rets.index[30 : 30 + len(picks)]
)
wealth = (1 + final_series).cumprod()

# --- 4. COLOR-CODED AUDIT ---
st.subheader("15-Day Strategy Audit")
audit_df = pd.DataFrame({
    "Date": final_series.tail(15).index.strftime('%Y-%m-%d'),
    "Ticker": picks[-15:],
    "Net Return": [f"{v:+.2%}" for v in final_series.tail(15).values]
})

def highlight_returns(val):
    num = float(val.strip('%').replace('+', ''))
    color = 'green' if num > 0 else 'red'
    return f'color: {color}; font-weight: bold;'

st.table(audit_df.style.applymap(highlight_returns, subset=['Net Return']))
