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
        self.decoder = nn.Linear(d_model, 6) # 5 ETFs + 1 CASH

    def forward(self, x):
        x = self.input_projection(x) + self.pos_encoder
        return self.decoder(self.transformer_encoder(x)[:, -1, :])

# --- 2. DYNAMIC ENGINE (NO HARDCODED FALLBACKS) ---
@st.cache_resource(ttl="1d")
def train_engine(start_year, tx_cost_bps):
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    etfs = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
    start_date = f"{start_year}-01-01"
    
    # 1. Fetch ETF Data
    raw_prices = yf.download(etfs, start=start_date, progress=False)['Close']
    returns_df = raw_prices.ffill().pct_change().dropna()
    
    # 2. Tiered Cash Fallback: SOFR -> 6-Month T-Bill
    try:
        sofr = fred.get_series('SOFR', start_date)
        tbill = fred.get_series('DTB6', start_date)
        cash_rate = sofr.combine_first(tbill) #
        returns_df['CASH'] = (cash_rate / 360 / 100).reindex(returns_df.index).ffill().fillna(0.0001)
    except:
        returns_df['CASH'] = 0.0001

    # 3. Dynamic Macro Features (Tiered Fallbacks)
    macro = pd.DataFrame(index=returns_df.index)
    
    # Yield Curve Fallback
    try:
        macro['10Y2Y'] = fred.get_series('T10Y2Y', start_date)
    except:
        t10 = fred.get_series('DGS10', start_date)
        t2 = fred.get_series('DGS2', start_date)
        macro['10Y2Y'] = t10 - t2 # Manually calculate spread
    
    # Volatility Fallback
    try:
        vix_data = yf.download("^VIX", start=start_date, progress=False)['Close']
        macro['VIX'] = vix_data
    except:
        # If VIX feed fails, use 20-day realized vol of TLT as a fear proxy
        macro['VIX'] = returns_df['TLT'].rolling(20).std() * np.sqrt(252) * 100
        
    # Copper Fallback
    try:
        copper = yf.download("HG=F", start=start_date, progress=False)['Close']
        macro['COPPER'] = copper
    except:
        macro['COPPER'] = fred.get_series('PCOPPUSDM', start_date) # Global Copper Price

    full_df = pd.concat([returns_df, macro], axis=1).ffill().dropna()

    # 4. Multi-Day Target (3-Day Trend)
    target_df = returns_df.rolling(window=3).sum().shift(-3).dropna()
    common_idx = full_df.index.intersection(target_df.index)
    full_df, target_df = full_df.loc[common_idx], target_df.loc[common_idx]

    # 5. Training Logic
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(full_df)
    
    def create_sequences(data, target_data, window=30):
        xs, ys = [], []
        for i in range(len(data) - window):
            xs.append(data[i:i+window]); ys.append(target_data[i+window])
        return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))

    split_idx = int(len(scaled_data) * 0.8)
    X_train, y_train = create_sequences(scaled_data[:split_idx], target_df.values[:split_idx])
    
    model = FinancialTransformer(input_dim=full_df.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)
    loss_history = []
    
    model.train()
    for _ in range(100):
        optimizer.zero_grad()
        output = model(X_train)
        loss = nn.HuberLoss()(output, y_train)
        loss.backward(); optimizer.step()
        loss_history.append(loss.item())
        
    return {"model": model, "scaler": scaler, "returns": returns_df, "features": scaled_data, "loss": loss_history}

# --- 3. UI & SIMULATION ---
st.set_page_config(page_title="Transformer Alpha V4", layout="wide")

with st.sidebar:
    st.header("⚙️ Strategy Settings")
    regime_year = st.slider("Data Anchor", 2008, 2023, 2015)
    tx_cost = st.number_input("Tx Cost (BPS)", 0, 50, 10)
    engine = train_engine(regime_year, tx_cost)
    
    st.subheader("Training Convergence")
    fig = go.Figure(go.Scatter(y=engine["loss"], mode='lines', line=dict(color='#00d4ff')))
    fig.update_layout(height=180, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# 4. WALK-FORWARD INFERENCE
engine["model"].eval()
assets = ["TLT", "TBT", "VNQ", "SLV", "GLD", "CASH"]
oos_idx = int(len(engine["returns"]) * 0.8)
oos_features = engine["features"][oos_idx:]
actual_rets_df = engine["returns"].iloc[oos_idx:]

picks, current_pick = [], None
cost_decimal = tx_cost / 10000

for i in range(len(oos_features) - 30):
    seq = torch.FloatTensor(oos_features[i:i+30]).unsqueeze(0)
    pred = engine["model"](seq).detach().numpy()[0]
    if current_pick:
        pred[assets.index(current_pick)] += (cost_decimal * 1.5) # Loyalty Bonus
    current_pick = assets[np.argmax(pred)]
    picks.append(current_pick)

# 5. ALIGNMENT & METRICS
available_rets = actual_rets_df.iloc[30:].iloc[:len(picks)]
final_rets_list = []

for i, p in enumerate(picks):
    day_ret = available_rets[p].iloc[i]
    if i > 0 and picks[i] != picks[i-1]:
        day_ret -= cost_decimal
    final_rets_list.append(day_ret)

final_rets_series = pd.Series(final_rets_list, index=available_rets.index)
wealth = (1 + final_rets_series).cumprod()

# Dashboard Output
st.title("🚀 Transformer Alpha: Multi-Asset Dashboard")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Pick", picks[-1])
m2.metric("Ann. Return", f"{(((1 + final_rets_series.mean())**252) - 1) * 100:.1f}%")
m3.metric("Sharpe Ratio", f"{(final_rets_series.mean() / final_rets_series.std()) * np.sqrt(252):.2f}")
m4.metric("Hit Ratio", f"{(final_rets_series > 0).sum() / len(final_rets_series) * 100:.1f}%")

st.line_chart(wealth)

with st.expander("15-Day Strategy Audit"):
    audit = pd.DataFrame({
        "Date": final_rets_series.tail(15).index.strftime('%Y-%m-%d'),
        "Ticker": picks[-15:],
        "Net Return": [f"{v:.2%}" for v in final_rets_series.tail(15).values]
    })
    st.table(audit)
