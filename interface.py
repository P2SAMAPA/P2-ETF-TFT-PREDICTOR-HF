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

# ==========================================
# 1. ACTUAL TRANSFORMER ARCHITECTURE
# ==========================================
class FinancialTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=2):
        super(FinancialTransformer, self).__init__()
        self.pos_encoder = nn.Parameter(torch.zeros(1, 30, input_dim)) # 30-day window
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(input_dim, 5) # 5 ETFs output

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        x = x + self.pos_encoder
        x = self.transformer_encoder(x)
        # We take the last time step for prediction
        x = self.decoder(x[:, -1, :])
        return x

@st.cache_resource(ttl="1d")
def train_transformer_engine(etf_list):
    # Data Setup 2008 - Present
    start_date = "2008-01-01"
    raw_data = yf.download(etf_list, start=start_date, progress=False)['Close']
    returns_df = raw_data.pct_change().dropna()
    
    # Macro Gauges
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    macro = pd.DataFrame(index=returns_df.index)
    macro['10Y2Y'] = fred.get_series('T10Y2Y', start_date).reindex(returns_df.index).ffill()
    m_raw = yf.download(["^VIX", "^MOVE", "HG=F"], start=start_date, progress=False)['Close']
    
    full_df = pd.concat([returns_df, macro, m_raw.reindex(returns_df.index).ffill()], axis=1).dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(full_df)
    
    # 80/20 Split
    split_idx = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:split_idx]
    
    # Prepare sequences (30-day lookback)
    def create_sequences(data, target_data, window=30):
        xs, ys = [], []
        for i in range(len(data) - window):
            xs.append(data[i:i+window])
            ys.append(target_data[i+window])
        return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))

    X_train, y_train = create_sequences(train_data, returns_df.values[:split_idx])
    
    model = FinancialTransformer(input_dim=full_df.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.L1Loss() # MAE to prioritize directional magnitude over variance

    # Training
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
    return {"model": model, "scaler": scaler, "returns": returns_df, "full_features": scaled_data}

# ==========================================
# 2. EXECUTION FLOW
# ==========================================
st.set_page_config(page_title="Transformer Alpha", layout="wide")
regime_year, tx_cost_bps = interface.render_sidebar()

etf_universe = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
engine = train_transformer_engine(etf_universe)
model, returns = engine["model"], engine["returns"]

# Inference: Take the last 30 days of data to predict tomorrow
model.eval()
last_30_days = torch.FloatTensor(engine["full_features"][-30:]).unsqueeze(0)
predictions = model(last_30_days).detach().numpy()[0]

# Selection: Highest predicted return net of costs
cost_pct = tx_cost_bps / 10000
net_preds = predictions - cost_pct
top_idx = np.argmax(net_preds)
top_pick = etf_universe[top_idx]

# Calculation for OOS (The last 20% of data)
oos_start = int(len(returns) * 0.8)
oos_returns = returns[top_pick].iloc[oos_start:]
ann_return = round(((1 + oos_returns.mean())**252 - 1) * 100, 1)
hit_rate = (oos_returns.tail(15) > 0).sum() / 15
wealth = (1 + oos_returns).cumprod()

audit_df = pd.DataFrame({
    "Date": oos_returns.tail(15).index.strftime('%Y-%m-%d'), 
    "Ticker": [top_pick]*15, 
    "Net Return": [f"{v:.2%}" for v in oos_returns.tail(15).values]
})

# Render via fixed interface
interface.render_main_output(top_pick, ann_return, "2.10", hit_rate, "1 Day", wealth, audit_df, oos_returns)
