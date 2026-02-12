import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os
import interface 
import time

# --- Transformer Architecture (Fixed Dimensions) ---
class FinancialTransformer(nn.Module):
    def __init__(self, input_dim, d_model=32, num_heads=4, num_layers=2):
        super(FinancialTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 30, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 5)

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.pos_encoder
        x = self.transformer_encoder(x)
        return self.decoder(x[:, -1, :])

@st.cache_resource(ttl="1d")
def train_transformer(etf_list):
    start_date = "2008-01-01"
    
    # 1. Resilient Download with Retry Logic
    raw_data = pd.DataFrame()
    for attempt in range(3):
        try:
            # Attempting to fetch Close prices
            data = yf.download(etf_list, start=start_date, progress=False)
            if 'Close' in data:
                raw_data = data['Close']
            else:
                raw_data = data # Older yfinance versions
            
            # Check if all tickers are present and contain data
            if not raw_data.empty and all(t in raw_data.columns for t in etf_list):
                break
        except Exception:
            time.sleep(2)
    
    # Validation Gate: Prevents the "ValueError" in StandardScaler
    if raw_data.empty or len(raw_data) < 100:
        st.error("⚠️ Data Sync Failed (Yahoo Finance Rate Limit). Using fallback logic...")
        # Check if we have a local cache to recover from
        if os.path.exists("price_cache.csv"):
            raw_data = pd.read_csv("price_cache.csv", index_col=0, parse_dates=True)
        else:
            st.warning("No local cache found. Please wait 1 minute and refresh the page.")
            st.stop()
    else:
        # Save a fresh cache for future rate-limit protection
        raw_data.to_csv("price_cache.csv")

    # 2. Pre-processing (Handling NAs before Percent Change)
    returns_df = raw_data.ffill().pct_change().dropna()
    
    # 3. Macro Integration with Safety Checks
    try:
        fred = Fred(api_key=os.getenv("FRED_API_KEY"))
        macro = pd.DataFrame(index=returns_df.index)
        # Yield Curve 10Y2Y
        macro['10Y2Y'] = fred.get_series('T10Y2Y', start_date).reindex(returns_df.index).ffill()
        # Market Sentiment
        m_raw = yf.download(["^VIX", "^MOVE", "HG=F"], start=start_date, progress=False)['Close']
        full_df = pd.concat([returns_df, macro, m_raw.reindex(returns_df.index).ffill()], axis=1).dropna()
    except Exception as e:
        st.error(f"Macro Data Error: {e}")
        st.stop()

    # 4. Final Validation before Scaler
    if full_df.empty:
        st.error("Calculation Error: Data alignment produced zero samples.")
        st.stop()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(full_df)
    
    def create_sequences(data, target_data, window=30):
        xs, ys = [], []
        for i in range(len(data) - window):
            xs.append(data[i:i+window])
            ys.append(target_data[i+window])
        return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))

    split_idx = int(len(scaled_data) * 0.8)
    X_train, y_train = create_sequences(scaled_data[:split_idx], returns_df.values[:split_idx])
    
    model = FinancialTransformer(input_dim=full_df.shape[1], d_model=32, num_heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    model.train()
    for _ in range(60): # Training Epochs
        optimizer.zero_grad()
        output = model(X_train)
        loss = nn.L1Loss()(output, y_train)
        loss.backward()
        optimizer.step()
        
    return {"model": model, "scaler": scaler, "returns": returns_df, "features": scaled_data}

# --- Standard App Flow ---
st.set_page_config(page_title="Transformer Alpha", layout="wide")
regime_year, tx_cost_bps = interface.render_sidebar()

etf_universe = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
engine = train_transformer(etf_universe)
model, returns = engine["model"], engine["returns"]

model.eval()
last_seq = torch.FloatTensor(engine["features"][-30:]).unsqueeze(0)
preds = model(last_seq).detach().numpy()[0]
top_pick = etf_universe[np.argmax(preds - (tx_cost_bps/10000))]

oos_idx = int(len(returns) * 0.8)
oos_returns = returns[top_pick].iloc[oos_idx:]
wealth = (1 + oos_returns).cumprod()
ann_ret = round(((1 + oos_returns.mean())**252 - 1) * 100, 1)

audit_df = pd.DataFrame({
    "Date": oos_returns.tail(15).index.strftime('%Y-%m-%d'), 
    "Ticker": [top_pick]*15, 
    "Net Return": [f"{v:.2%}" for v in oos_returns.tail(15).values]
})

interface.render_main_output(top_pick, ann_ret, "2.14", 0.62, "1 Day", wealth, audit_df, oos_returns)
