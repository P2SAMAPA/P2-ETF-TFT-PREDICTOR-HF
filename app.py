import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import os

# ==========================================
# 1. ADVANCED DL MODELS (VAE + LSTM)
# ==========================================
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=3):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, latent_dim * 2))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 16), nn.ReLU(), nn.Linear(16, input_dim), nn.Sigmoid())

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decoder(z)

class SuperLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super(SuperLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, 64, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==========================================
# 2. DATA ENGINE (Calculated Spreads + Macro)
# ==========================================
def get_unified_data(start_year, etf_list):
    start_date = f"{start_year}-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # ETF Prices
    prices = yf.download(etf_list, start=start_date, end=end_date, progress=False)['Close']
    
    # Macro Spreads (Fixed T10Y5Y issue)
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    macro = pd.DataFrame(index=prices.index)
    macro['10Y2Y'] = fred.get_series('T10Y2Y', start_date, end_date)
    
    # Manual Calculation for 10Y-5Y
    dgs10 = fred.get_series('DGS10', start_date, end_date)
    dgs5 = fred.get_series('DGS5', start_date, end_date)
    macro['10Y5Y'] = dgs10 - dgs5
    
    # Volatility & Ratios
    m_tickers = {"^MOVE": "MOVE", "^SKEW": "SKEW", "^PCCR": "PCC", "GC=F": "Gold", "HG=F": "Copper"}
    m_raw = yf.download(list(m_tickers.keys()), start=start_date, end=end_date, progress=False)['Close'].rename(columns=m_tickers)
    m_raw['Au_Cu'] = m_raw['Gold'] / m_raw['Copper']
    
    return pd.concat([prices, macro, m_raw[['MOVE', 'SKEW', 'PCC', 'Au_Cu']]], axis=1).ffill().dropna()

# ==========================================
# 3. AUTO-TRAINER (Every 3 Days)
# ==========================================
@st.cache_resource(ttl="3d")
def run_super_engine(start_year, etf_list):
    full_df = get_unified_data(start_year, etf_list)
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(full_df)
    
    # Divide data: ETFs (0-8), Macro (9-end)
    etf_count = len(etf_list)
    macro_scaled = torch.FloatTensor(scaled[:, etf_count:])
    
    # 1. Denoise with VAE
    vae = VAE(macro_scaled.shape[1])
    optimizer_v = torch.optim.Adam(vae.parameters(), lr=0.01)
    for _ in range(50):
        recon = vae(macro_scaled)
        loss = nn.MSELoss()(recon, macro_scaled)
        optimizer_v.zero_grad()
        loss.backward()
        optimizer_v.step()
    
    denoised_macro = vae(macro_scaled).detach().numpy()
    final_features = np.hstack([scaled[:, :etf_count], denoised_macro])
    
    # 2. Predict with LSTM
    window = 30
    X, y = [], []
    for i in range(len(final_features)-window):
        X.append(final_features[i:i+window])
        y.append(final_features[i+window, :etf_count])
        
    lstm = SuperLSTM(final_features.shape[1], etf_count)
    optimizer_l = torch.optim.Adam(lstm.parameters(), lr=0.001)
    for _ in range(30):
        pred = lstm(torch.FloatTensor(np.array(X)))
        loss = nn.MSELoss()(pred, torch.FloatTensor(np.array(y)))
        optimizer_l.zero_grad(); loss.backward(); optimizer_l.step()
        
    return {"model": lstm, "vae": vae, "scaler": scaler, "data": full_df, "features": final_features}

# ==========================================
# 4. DASHBOARD UI
# ==========================================
st.set_page_config(page_title="Institutional Alpha Engine", layout="wide")
st.title("🏛️ Institutional Alpha Engine")

# Controls
st.sidebar.header("Regime & Strategy")
regime_year = st.sidebar.select_slider("Data Anchor (History)", options=[2008, 2015, 2019, 2021])
tx_cost_bps = st.sidebar.slider("Transaction Cost (bps)", 0, 100, 15)

etf_universe = ["VCLT", "TLT", "TBT", "MBB", "VNQ", "HYG", "SLV", "GLD", "PFF"]

# Run Engine
with st.spinner("Analyzing Market Regimes..."):
    engine = run_super_engine(regime_year, etf_universe)

# Inference
model = engine["model"]
scaler = engine["scaler"]
features = engine["features"]
current_prices = engine["data"][etf_universe].iloc[-1].values

model.eval()
last_window = torch.FloatTensor(features[-30:].reshape(1, 30, -1))
raw_pred = model(last_window).detach().numpy()
# Reverse scaling for prices
pred_prices = scaler.inverse_transform(np.hstack([raw_pred, np.zeros((1, features.shape[1]-len(etf_universe)))]))[0][:len(etf_universe)]

# Output Calculation
results = []
cost = tx_cost_bps / 10000
for i, t in enumerate(etf_universe):
    ret_base = (pred_prices[i] - current_prices[i]) / current_prices[i]
    r1, r3, r5 = ret_base-cost, (ret_base*1.4)-cost, (ret_base*1.8)-cost
    best_ret = max(r1, r3, r5)
    period = "1D" if best_ret == r1 else "3D" if best_ret == r3 else "5D"
    results.append({"ETF": t, "Expected Net Return": best_ret, "Optimal Horizon": period})

res_df = pd.DataFrame(results).sort_values("Expected Net Return", ascending=False)

# UI Display
c1, c2 = st.columns([1, 2])
c1.metric("Top Alpha Signal", res_df.iloc[0]['ETF'], f"{res_df.iloc[0]['Expected Net Return']:.2%}")
c1.write(f"**Recommended Holding:** {res_df.iloc[0]['Optimal Horizon']}")
c2.table(res_df.style.format({"Expected Net Return": "{:.2%}"}))

st.plotly_chart(go.Figure(data=[go.Scatter(x=engine['data'].index[-100:], y=engine['data'][t].tail(100), name=t) for t in etf_universe]), width='stretch')
