import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os
import gc
import pytz
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download, HfApi
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, nearest_workday,
    USMartinLutherKingJr, USPresidentsDay, GoodFriday,
    USMemorialDay, USLaborDay, USThanksgivingDay
)

# --- 1. CONFIG & HF PERSISTENCE ---
REPO_ID = "P2SAMAPA/etf-alpha-data" 
FILENAME = "historical_cache.parquet"
HF_TOKEN = os.getenv("HF_TOKEN")

def sync_data_persistent(assets, start_date_str):
    if not HF_TOKEN:
        return yf.download(assets, start=start_date_str, progress=False)['Close']
    try:
        path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset", token=HF_TOKEN)
        df = pd.read_parquet(path)
        last_date = df.index.max()
    except Exception:
        df = yf.download(assets, start=start_date_str, progress=False)['Close']
        save_to_hf(df)
        return df
    
    yesterday = datetime.now() - timedelta(days=1)
    if last_date.date() < yesterday.date():
        delta_start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        new_data = yf.download(assets, start=delta_start, progress=False)['Close']
        if not new_data.empty:
            df = pd.concat([df, new_data]).drop_duplicates().sort_index()
            save_to_hf(df)
    return df

def save_to_hf(df):
    df.to_parquet(FILENAME)
    api = HfApi()
    api.upload_file(path_or_fileobj=FILENAME, path_in_repo=FILENAME, repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)

# --- 2. NYSE CALENDAR ---
class USTradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday('Juneteenth', month=6, day=19, start_date='2021-06-18', observance=nearest_workday),
        Holiday('USIndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]

def get_market_execution_date():
    tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(tz)
    inst = USTradingCalendar()
    holidays = inst.holidays(start=now_et.date(), end=now_et.date() + timedelta(days=1))
    if now_et.strftime('%Y-%m-%d') not in holidays and now_et.weekday() < 5 and now_et.hour < 16:
        return now_et.strftime('%B %d, %Y')
    curr = now_et + timedelta(days=1)
    all_hols = inst.holidays(start=now_et.date(), end=now_et.date() + timedelta(days=14))
    while True:
        if curr.weekday() < 5 and curr.strftime('%Y-%m-%d') not in all_hols:
            return curr.strftime('%B %d, %Y')
        curr += timedelta(days=1)

# --- 3. MODEL CORE (RAM OPTIMIZED) ---
class MomentumTransformer(nn.Module):
    def __init__(self, input_dim, d_model=32, num_heads=4, seq_len=30):
        super(MomentumTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
        self.decoder = nn.Linear(d_model, 6) 

    def forward(self, x):
        x = self.input_projection(x) + self.pos_encoder
        x = self.transformer_encoder(x)
        return self.decoder(x[:, -1, :])

@st.cache_resource(ttl="1h")
def train_engine(start_year, tx_cost_bps):
    gc.collect() # Force clear RAM
    print(f">>> Training {start_year} with Memory Guard...")
    
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    etfs = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
    start_date = f"{start_year}-01-01"
    
    data = sync_data_persistent(etfs, start_date)
    returns_df = data.ffill().pct_change().fillna(0)
    
    try:
        sofr = fred.get_series('SOFR', start_date)
        returns_df['CASH'] = (sofr / 360 / 100).reindex(returns_df.index).ffill().fillna(0.0001)
    except:
        returns_df['CASH'] = 0.0001

    for asset in etfs + ['CASH']:
        returns_df[f'{asset}_ROC_10'] = returns_df[asset].pct_change(10)

    vix = yf.download("^VIX", start=start_date, progress=False)['Close']
    returns_df['VIX'] = vix.reindex(returns_df.index).ffill().fillna(20)
    
    target_df = returns_df[etfs + ['CASH']].rolling(3).sum().shift(-3).dropna()
    features_df = returns_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    common_idx = features_df.index.intersection(target_df.index)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features_df.loc[common_idx].astype(np.float64))
    
    split_idx = int(len(scaled_data) * 0.8)
    X = []
    for i in range(len(scaled_data)-30): X.append(scaled_data[i:i+30])
    X_train = torch.FloatTensor(np.array(X[:split_idx]))
    y_train = torch.FloatTensor(target_df.values[30:split_idx+30])

    model = MomentumTransformer(input_dim=features_df.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 40 if start_year >= 2020 else 75 # Speed optimization
    model.train()
    for e in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = nn.MSELoss()(output, y_train)
        loss.backward(); optimizer.step()
        if e % 10 == 0: print(f"Progress: {e}/{epochs}")
        
    print(">>> Training Complete.")
    return {"model": model, "scaler": scaler, "returns": returns_df[etfs+['CASH']], "features": scaled_data, "vix": returns_df['VIX']}

# --- 4. DASHBOARD UI ---
st.set_page_config(page_title="ETF Alpha Maximizer", layout="wide")

with st.sidebar:
    st.header("⚙️ Strategy Settings")
    regime_year = st.slider("Year Anchor", 2008, 2023, 2008)
    tx_cost = st.number_input("Tx Cost (BPS)", 0, 50, 15)
    engine = train_engine(regime_year, tx_cost)

# Run Inference
engine["model"].eval()
assets = ["TLT", "TBT", "VNQ", "SLV", "GLD", "CASH"]
split_point = int(len(engine["features"])*0.8)
oos_features = engine["features"][split_point:]
actual_rets = engine["returns"].iloc[split_point:]

picks = []
for i in range(len(oos_features)-30):
    with torch.no_grad():
        pred = engine["model"](torch.FloatTensor(oos_features[i:i+30]).unsqueeze(0)).numpy()[0]
    picks.append(assets[np.argmax(pred)])

# Wealth Calculation
final_series = pd.Series([actual_rets[p].iloc[i+30] for i, p in enumerate(picks)], index=actual_rets.index[30:30+len(picks)])
wealth = (1 + final_series).cumprod()

# Layout
st.title("Fixed Income/Commodity ETF Alpha Maximizer")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"**Current Pick**\n<h2 style='margin:0;'>{picks[-1]}</h2><p style='font-size:12px; color:gray;'>Execution: {get_market_execution_date()}</p>", unsafe_allow_html=True)
with m2:
    ann_ret = (((1 + final_series.mean())**252) - 1) * 100
    st.markdown(f"**Ann. Return**\n<h2 style='margin:0;'>{ann_ret:.1f}%</h2><p style='font-size:12px; color:gray;'>OOS Results</p>", unsafe_allow_html=True)
with m3:
    sharpe = (final_series.mean() / final_series.std()) * np.sqrt(252)
    st.markdown(f"**Sharpe Ratio**\n<h2 style='margin:0;'>{sharpe:.2f}</h2><p style='font-size:12px; color:gray;'>Risk-Adjusted</p>", unsafe_allow_html=True)
with m4:
    hit_ratio = (final_series > 0).sum() / len(final_series) * 100
    st.markdown(f"**Hit Ratio**\n<h2 style='margin:0;'>{hit_ratio:.1f}%</h2><p style='font-size:12px; color:gray;'>Success Rate</p>", unsafe_allow_html=True)

st.line_chart(wealth)
