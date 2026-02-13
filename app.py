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
import time
from datetime import datetime, timedelta
import pytz
from huggingface_hub import hf_hub_download, HfApi
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, nearest_workday,
    USMartinLutherKingJr, USPresidentsDay, GoodFriday,
    USMemorialDay, USLaborDay, USThanksgivingDay
)

# --- 1. CONFIG & HF PERSISTENCE ---
# Change these to match your Hugging Face Setup
REPO_ID = "your-username/etf-alpha-data" 
FILENAME = "historical_cache.parquet"
HF_TOKEN = os.getenv("HF_TOKEN")

def sync_data_persistent(assets, start_date_str):
    """Downloads only missing data and saves to HF Dataset."""
    if not HF_TOKEN:
        st.error("HF_TOKEN secret not found in Space Settings.")
        return yf.download(assets, start=start_date_str, progress=False)['Close']

    api = HfApi()
    df = None
    
    # Try to load existing cache from HF
    try:
        path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset", token=HF_TOKEN)
        df = pd.read_parquet(path)
        last_date = df.index.max()
    except Exception as e:
        st.warning("No HF cache found. Performing full initial pull...")
        df = yf.download(assets, start=start_date_str, progress=False)['Close']
        save_to_hf(df)
        return df

    # Check for incremental updates (if today > last saved date)
    yesterday = datetime.now() - timedelta(days=1)
    if last_date.date() < yesterday.date():
        # Only pull the 'delta' (from last date to now)
        delta_start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        new_data = yf.download(assets, start=delta_start, progress=False)['Close']
        
        if not new_data.empty:
            df = pd.concat([df, new_data]).drop_duplicates().sort_index()
            save_to_hf(df)
            st.sidebar.success(f"Synced {len(new_data)} new market days to HF storage.")
    
    return df

def save_to_hf(df):
    """Saves the local dataframe as a parquet and pushes to HF."""
    df.to_parquet(FILENAME)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=FILENAME,
        path_in_repo=FILENAME,
        repo_id=REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )

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
    is_holiday = now_et.strftime('%Y-%m-%d') in holidays
    is_weekend = now_et.weekday() >= 5
    if not is_holiday and not is_weekend and now_et.hour < 16:
        return now_et.strftime('%B %d, %Y')
    curr = now_et + timedelta(days=1)
    all_holidays = inst.holidays(start=now_et.date(), end=now_et.date() + timedelta(days=14))
    while True:
        if curr.weekday() < 5 and curr.strftime('%Y-%m-%d') not in all_holidays:
            return curr.strftime('%B %d, %Y')
        curr += timedelta(days=1)

# --- 3. MODEL ARCHITECTURE ---
class MomentumTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, num_heads=8, seq_len=30):
        super(MomentumTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        mask = torch.arange(seq_len).float()
        self.register_buffer('time_mask', torch.exp(mask - seq_len + 1).unsqueeze(0).unsqueeze(0))
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)
        self.decoder = nn.Linear(d_model, 6) 

    def forward(self, x):
        x = self.input_projection(x) * self.time_mask.transpose(1, 2) + self.pos_encoder
        x = self.transformer_encoder(x)
        return self.decoder(x[:, -1, :])

def add_momentum_indicators(df, assets):
    for asset in assets:
        if asset == 'CASH' or asset not in df.columns: continue
        df[f'{asset}_ROC_3'] = df[asset].pct_change(3)
        df[f'{asset}_ROC_10'] = df[asset].pct_change(10)
        ema = df[asset].ewm(span=20).mean()
        df[f'{asset}_Dist_EMA'] = (df[asset] - ema) / (ema + 1e-9)
    return df

@st.cache_resource(ttl="1d")
def train_engine(start_year, tx_cost_bps):
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    etfs = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
    start_date = f"{start_year}-01-01"
    
    # NEW PERSISTENT SYNC LOGIC
    data = sync_data_persistent(etfs, start_date)
    
    returns_df = data.ffill().pct_change().fillna(0)
    try:
        sofr = fred.get_series('SOFR', start_date)
        returns_df['CASH'] = (sofr / 360 / 100).reindex(returns_df.index).ffill().fillna(0.0001)
    except:
        returns_df['CASH'] = 0.0001

    full_df = add_momentum_indicators(returns_df.copy(), etfs + ['CASH'])
    vix = yf.download("^VIX", start=start_date, progress=False)['Close']
    full_df['VIX'] = vix.reindex(full_df.index).ffill().fillna(20)
        
    target_df = returns_df.rolling(3).sum().shift(-3).dropna()
    full_df = full_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    common_idx = full_df.index.intersection(target_df.index)
    full_df, target_df = full_df.loc[common_idx], target_df.loc[common_idx]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(full_df.astype(np.float64))
    
    split_idx = int(len(scaled_data) * 0.8)
    def create_seq(data, target, window=30):
        xs, ys = [], []
        for i in range(len(data)-window):
            xs.append(data[i:i+window]); ys.append(target[i+window])
        return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))

    X_train, y_train = create_seq(scaled_data[:split_idx], target_df.values[:split_idx])
    
    model = MomentumTransformer(input_dim=full_df.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_history = []
    
    model.train()
    for _ in range(100):
        optimizer.zero_grad()
        output = model(X_train)
        weights = torch.ones_like(y_train)
        winners = torch.argmax(y_train, dim=1)
        for i, w_idx in enumerate(winners): weights[i, w_idx] = 2.5
        loss = (weights * (output - y_train)**2).mean()
        loss.backward(); optimizer.step()
        loss_history.append(loss.item())
        
    return {"model": model, "scaler": scaler, "returns": returns_df, "features": scaled_data, "loss": loss_history, "vix": full_df['VIX']}

# --- 4. UI ---
st.set_page_config(page_title="ETF Alpha Maximizer", layout="wide")

with st.sidebar:
    st.header("⚙️ Strategy Settings")
    regime_year = st.slider("Year Anchor", 2008, 2023, 2008)
    tx_cost = st.number_input("Tx Cost (BPS)", 0, 50, 15)
    engine = train_engine(regime_year, tx_cost)
    
    st.subheader("Training Convergence")
    fig_loss = go.Figure(go.Scatter(y=engine["loss"], mode='lines', line=dict(color='#00d4ff')))
    fig_loss.update_layout(height=150, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark")
    st.plotly_chart(fig_loss, use_container_width=True)

engine["model"].eval()
assets = ["TLT", "TBT", "VNQ", "SLV", "GLD", "CASH"]
oos_idx = int(len(engine["returns"]) * 0.8)
oos_features = engine["features"][oos_idx:]
actual_rets = engine["returns"].iloc[oos_idx:]
vix_oos = engine["vix"].iloc[oos_idx:]

picks, current_pick, cost_dec = [], None, tx_cost/10000
for i in range(len(oos_features)-30):
    with torch.no_grad():
        pred = engine["model"](torch.FloatTensor(oos_features[i:i+30]).unsqueeze(0)).numpy()[0]
    loyalty_mult = 1.5 if vix_oos.iloc[i+30] < 25 else 0.5
    if current_pick: pred[assets.index(current_pick)] += (cost_dec * loyalty_mult)
    current_pick = assets[np.argmax(pred)]
    picks.append(current_pick)

final_series = pd.Series([actual_rets[p].iloc[i+30] for i, p in enumerate(picks)], index=actual_rets.index[30:30+len(picks)])
wealth = (1 + final_series).cumprod()
trade_date = get_market_execution_date()

st.title("Fixed Income/Commodity ETF Alpha Maximizer")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"**Current Pick**\n<h2 style='margin:0;'>{picks[-1]}</h2><p style='font-size:12px; color:gray;'>Execution: {trade_date}</p>", unsafe_allow_html=True)
with m2:
    st.markdown(f"**Ann. Return**\n<h2 style='margin:0;'>{(((1 + final_series.mean())**252) - 1) * 100:.1f}%</h2><p style='font-size:12px; color:gray;'>{len(final_series)//21} OOS Months</p>", unsafe_allow_html=True)
with m3:
    st.markdown(f"**Sharpe Ratio**\n<h2 style='margin:0;'>{(final_series.mean() / final_series.std()) * np.sqrt(252):.2f}</h2><p style='font-size:12px; color:gray;'>Risk-Adjusted</p>", unsafe_allow_html=True)
with m4:
    st.markdown(f"**Hit Ratio**\n<h2 style='margin:0;'>{(final_series > 0).sum() / len(final_series) * 100:.1f}%</h2><p style='font-size:12px; color:gray;'>15-Day Strategy Window</p>", unsafe_allow_html=True)

st.subheader("Out-of-Sample Performance")
st.line_chart(wealth)

st.subheader("15-Day Strategy Audit")
audit_df = pd.DataFrame({"Date": final_series.tail(15).index.strftime('%Y-%m-%d'), "Ticker": picks[-15:], "Net Return": [f"{v:+.2%}" for v in final_series.tail(15).values]})
st.table(audit_df.style.applymap(lambda v: f"color: {'green' if '+' in v else 'red'}; font-weight: bold;", subset=['Net Return']))

st.divider()
st.header("📘 Model Methodology & Algorithm Details")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### **Architecture: Momentum Transformer**
    * **Self-Attention Mechanism:** Maps non-linear dependencies across a 30-day lookback.
    * **Recency Biasing:** Exponential time-decay mask favors the last 72 hours of price action.
    """)
with col2:
    st.markdown("""
    ### **Persistent Storage & Optimization**
    * **Delta-Sync Memory:** Leverages a Hugging Face Dataset as a permanent Parquet store, reducing API pings by only downloading new trading days.
    * **Execution:** Integrated NYSE trading calendar ensures trade timing matches US market availability.
    """)
