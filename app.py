import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import plotly.graph_objects as go
import os

# ==========================================
# 1. ADVANCED DL ENSEMBLE (VAE + LSTM)
# ==========================================
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=4):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z)

class SuperLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super(SuperLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==========================================
# 2. DATA PIPELINE (FIXED 2026 TICKERS)
# ==========================================
def get_unified_data(start_year, etf_list):
    start_date = f"{start_year}-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Core ETF Prices
    prices = yf.download(etf_list, start=start_date, end=end_date, progress=False)['Close']
    
    # Fixed FRED Spreads
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    macro = pd.DataFrame(index=prices.index)
    macro['10Y2Y'] = fred.get_series('T10Y2Y', start_date, end_date)
    
    # Manual 10Y5Y Spread Calculation
    dgs10 = fred.get_series('DGS10', start_date, end_date)
    dgs5 = fred.get_series('DGS5', start_date, end_date)
    macro['10Y5Y'] = dgs10 - dgs5
    
    # Fixed Market Signals (VIX replaces broken PCCR)
    m_tickers = {"^VIX": "VIX", "^MOVE": "MOVE", "^SKEW": "SKEW", "GC=F": "Gold", "HG=F": "Copper"}
    m_raw = yf.download(list(m_tickers.keys()), start=start_date, end=end_date, progress=False)['Close'].rename(columns=m_tickers)
    m_raw['Au_Cu_Ratio'] = m_raw['Gold'] / m_raw['Copper']
    
    combined = pd.concat([prices, macro, m_raw[['VIX', 'MOVE', 'SKEW', 'Au_Cu_Ratio']]], axis=1)
    return combined.ffill().dropna()

# ==========================================
# 3. AUTO-TRAINING (Every 3 Days)
# ==========================================
@st.cache_resource(ttl="3d", show_spinner="Retraining DL Ensemble for current market regime...")
def run_super_engine(start_year, etf_list):
    full_df = get_unified_data(start_year, etf_list)
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(full_df)
    
    etf_count = len(etf_list)
    macro_scaled = torch.FloatTensor(scaled[:, etf_count:])
    
    # VAE Denoising
    vae = VAE(macro_scaled.shape[1])
    opt_v = torch.optim.Adam(vae.parameters(), lr=0.01)
    for _ in range(50):
        recon = vae(macro_scaled)
        loss = nn.MSELoss()(recon, macro_scaled)
        opt_v.zero_grad(); loss.backward(); opt_v.step()
    
    denoised_macro = vae(macro_scaled).detach().numpy()
    final_features = np.hstack([scaled[:, :etf_count], denoised_macro])
    
    # LSTM Prediction Training
    window = 30
    X, y = [], []
    for i in range(len(final_features) - window):
        X.append(final_features[i:i+window])
        y.append(final_features[i+window, :etf_count])
        
    model = SuperLSTM(final_features.shape[1], etf_count)
    opt_l = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(40):
        pred = model(torch.FloatTensor(np.array(X)))
        loss = nn.MSELoss()(pred, torch.FloatTensor(np.array(y)))
        opt_l.zero_grad(); loss.backward(); opt_l.step()
        
    return {"model": model, "scaler": scaler, "data": full_df, "features": final_features, "timestamp": datetime.now()}

# ==========================================
# 4. DASHBOARD UI
# ==========================================
st.set_page_config(page_title="Alpha Super-Engine", layout="wide")
st.title("🏛️ Institutional Alpha Super-Engine")

# Sidebar
st.sidebar.header("Global Constraints")
regime_year = st.sidebar.select_slider("Historical Anchor (Regime)", options=[2008, 2015, 2019, 2021], value=2015)
tx_cost_bps = st.sidebar.slider("Transaction Cost (bps)", 0, 100, 15)

etf_universe = ["VCLT", "TLT", "TBT", "MBB", "VNQ", "HYG", "SLV", "GLD", "PFF"]

# Engine Execution
engine = run_super_engine(regime_year, etf_universe)
st.sidebar.success(f"Engine Fresh: Trained {engine['timestamp'].strftime('%m/%d %H:%M')}")

# Optimization Logic
model = engine["model"]
scaler = engine["scaler"]
features = engine["features"]
model.eval()

# Forecast
last_window = torch.FloatTensor(features[-30:].reshape(1, 30, -1))
raw_pred = model(last_window).detach().numpy()
# Reverse-Scale prices only
dummy = np.zeros((1, features.shape[1]))
dummy[0, :len(etf_universe)] = raw_pred
pred_prices = scaler.inverse_transform(dummy)[0][:len(etf_universe)]

current_prices = engine["data"][etf_universe].iloc[-1].values
expected_rets = (pred_prices - current_prices) / current_prices

# Ranking Table
results = []
cost_pct = tx_cost_bps / 10000
for i, t in enumerate(etf_universe):
    r1, r3, r5 = expected_rets[i]-cost_pct, (expected_rets[i]*1.4)-cost_pct, (expected_rets[i]*1.8)-cost_pct
    best_val = max(r1, r3, r5)
    period = "1 Day" if best_val == r1 else "3 Days" if best_val == r3 else "5 Days"
    results.append({"ETF": t, "Expected Net Return": best_val, "Optimal Horizon": period})

res_df = pd.DataFrame(results).sort_values("Expected Net Return", ascending=False)

# UI Layout
c1, c2 = st.columns([1, 2])
with c1:
    st.metric("Top Alpha Signal", res_df.iloc[0]['ETF'], f"{res_df.iloc[0]['Expected Net Return']:.2%}")
    st.info(f"Recommended Holding: **{res_df.iloc[0]['Optimal Horizon']}**")
    st.write("---")
    st.write(f"**Transaction Cost Impact:** {tx_cost_bps} bps deducted from raw forecast.")

with c2:
    st.subheader("Opportunity Ranking (Net of Costs)")
    st.dataframe(res_df.style.format({"Expected Net Return": "{:.2%}"}), use_container_width=True)

# Performance Visualization
st.divider()
fig = go.Figure()
for t in etf_universe:
    fig.add_trace(go.Scatter(x=engine['data'].index[-120:], y=engine['data'][t].tail(120), name=t))
fig.update_layout(title="ETF Momentum Context (Last 120 Days)", template="plotly_dark", height=400)
st.plotly_chart(fig, use_container_width=True)
