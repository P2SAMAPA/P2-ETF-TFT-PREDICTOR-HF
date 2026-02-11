import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from polygon import RESTClient
from fredapi import Fred
import yfinance as yf
import os
import time
from datetime import datetime, timedelta

# ==========================================
# 1. API KEY INITIALIZATION (Fail-Safe)
# ==========================================
# Using os.getenv to prevent StreamlitSecretNotFoundError in Docker
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

# Fallback for local testing via secrets.toml
if not POLYGON_API_KEY:
    try:
        POLYGON_API_KEY = st.secrets.get("POLYGON_API_KEY")
        FRED_API_KEY = st.secrets.get("FRED_API_KEY")
    except:
        pass

# UI Guard: If keys are missing, stop the app and show instructions
if not POLYGON_API_KEY or not FRED_API_KEY:
    st.set_page_config(page_title="API Setup Required", page_icon="🔑")
    st.error("### 🔑 API Keys Missing")
    st.info("""
    Please add your API keys to the **Hugging Face Space Secrets**:
    1. Go to **Settings** tab.
    2. Scroll to **Variables and secrets**.
    3. Add `POLYGON_API_KEY` and `FRED_API_KEY`.
    4. Click **Factory Reboot** at the bottom of the Settings page.
    """)
    st.stop()

# Initialize Clients
polygon_client = RESTClient(POLYGON_API_KEY)
fred = Fred(api_key=FRED_API_KEY)

# ==========================================
# 2. DATA FETCHING (Polygon with yfinance Fallback)
# ==========================================
@st.cache_data(ttl=3600)
def get_hybrid_data(ticker_list, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    end_str = end_date.strftime('%Y-%m-%d')
    start_str = start_date.strftime('%Y-%m-%d')
    
    combined_data = pd.DataFrame()
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, ticker in enumerate(ticker_list):
        success = False
        status_text.text(f"Fetching {ticker}...")
        
        # Try Polygon First
        try:
            aggs = polygon_client.get_aggs(ticker, 1, "day", start_str, end_str)
            df = pd.DataFrame(aggs)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            combined_data[ticker] = df['close']
            success = True
        except Exception as e:
            # If Polygon 429 Rate Limit hit, use yfinance
            st.warning(f"Polygon limit/error for {ticker}. Trying yfinance...")
        
        # Fallback to yfinance
        if not success:
            try:
                yf_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not yf_data.empty:
                    # Handle multi-index columns if necessary
                    if isinstance(yf_data.columns, pd.MultiIndex):
                        combined_data[ticker] = yf_data['Close'][ticker]
                    else:
                        combined_data[ticker] = yf_data['Close']
                    success = True
            except Exception as yf_e:
                st.error(f"Could not fetch {ticker} from any source.")

        progress_bar.progress((i + 1) / len(ticker_list))
    
    status_text.text("Data Retrieval Complete.")
    return combined_data

# ==========================================
# 3. APP UI
# ==========================================
st.set_page_config(page_title="Fixed Income ETF Maximizer", layout="wide")

st.title("📈 Fixed Income ETF Return Maximizer")
st.markdown("Optimization using LSTM & Economic Indicators (FRED + Polygon + yfinance)")

# Sidebar Configuration
st.sidebar.header("Parameters")
lookback_days = st.sidebar.slider("Historical Lookback (Days)", 365, 3650, 1825)

# Your requested ETF list
fixed_income_etfs = ["VCLT", "TLT", "TBT", "MBB", "VNQ", "HYG", "SLV", "GLD", "PFF"]

selected_tickers = st.sidebar.multiselect(
    "Select ETFs", 
    fixed_income_etfs, 
    default=["VCLT", "TLT", "HYG"]
)

if selected_tickers:
    data = get_hybrid_data(selected_tickers, lookback_days)
    
    if not data.empty:
        st.subheader("Historical Performance")
        fig = go.Figure()
        for t in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[t], name=t))
        
        # Using 2026 syntax: width='stretch' replaces use_container_width=True
        st.plotly_chart(fig, width='stretch')
        
        st.success("Data loaded successfully! LSTM Model ready for training.")
        
        # Display raw data for verification
        with st.expander("View Raw Data Table"):
            st.dataframe(data, width='stretch')
    else:
        st.error("No data found for the selected tickers.")
else:
    st.info("Please select at least one ETF from the sidebar to begin.")
