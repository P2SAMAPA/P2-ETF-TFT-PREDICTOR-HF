# app.py - FINAL VERSION (push this exact file to main)

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta
from polygon import RESTClient
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message="Downcasting object dtype arrays")

ETFS = ['TLT', 'VCLT', 'TBT', 'VNQ', 'SLV', 'GLD', 'PFF', 'MBB']

SIGNAL_TICKERS = [
    '^VIX', 'MOVE', 'IEF', 'SHY', 'HYG', 'HG=F'
]

ALL_TICKERS = ETFS + SIGNAL_TICKERS

@st.cache_data(ttl=3600)
def fetch_data(start_date: str, end_date: str) -> pd.DataFrame:
    client = RESTClient()
    data = {}
    for ticker in ALL_TICKERS:
        try:
            aggs = client.get_aggs(ticker, 1, "day", start_date, end_date, limit=50000)
            df = pd.DataFrame(aggs)
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
            df = df.set_index('date')[['close']].rename(columns={'close': ticker})
            data[ticker] = df.astype(float)
        except Exception as e:
            st.warning(f"Failed to fetch {ticker}: {e}")
            data[ticker] = pd.Series(dtype=float, name=ticker)

    combined = pd.concat(data.values(), axis=1).sort_index()
    combined.index = pd.to_datetime(combined.index)
    return combined.astype(float)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().astype(float)

    if 'IEF' in df.columns and 'SHY' in df.columns:
        df['10Y_2Y_proxy'] = df['IEF'] - df['SHY']

    if 'HYG' in df.columns and 'IEF' in df.columns:
        df['Credit_Spread_proxy'] = df['HYG'] - df['IEF']

    if 'GLD' in df.columns and 'HG=F' in df.columns:
        df['Gold_Copper_ratio'] = df['GLD'] / df['HG=F']

    for col in list(df.columns):
        if col in ETFS + ['^VIX', 'MOVE']:
            for lag in [1, 3, 5, 10]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
            df[f'{col}_ma5']  = df[col].rolling(5).mean()
            df[f'{col}_ma20'] = df[col].rolling(20).mean()

    df = df.ffill().bfill()
    return df.dropna(how='all')

# (The rest of the file — LSTM classes, train_lstm, run_backtest, and Streamlit UI — remains unchanged from the version I gave you previously)

# Paste the remaining code (from class TimeSeriesDataset onward) exactly as in my previous full response
