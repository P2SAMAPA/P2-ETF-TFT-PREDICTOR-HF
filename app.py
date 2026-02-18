import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
from huggingface_hub import HfApi
try:
    import pandas_datareader.data as web
except ImportError:
    st.error("Missing pandas_datareader. Please add it to requirements.txt")
    st.stop()
from datetime import datetime, timedelta
import pytz
try:
    import pandas_market_calendars as mcal
    NYSE_CALENDAR_AVAILABLE = True
except ImportError:
    NYSE_CALENDAR_AVAILABLE = False
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
try:
    import xgboost as xgb
except ImportError:
    st.error("Missing xgboost. Please install: pip install xgboost")
    st.stop()
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------
# 1. CORE CONFIG & SYNC WINDOW
# ------------------------------
REPO_ID = "P2SAMAPA/my-etf-data"

def get_next_trading_day(current_date):
    """Get next valid NYSE trading day (skip weekends and holidays)"""
    if NYSE_CALENDAR_AVAILABLE:
        try:
            nyse = mcal.get_calendar('NYSE')
            # Get next 10 days of trading schedule
            schedule = nyse.schedule(
                start_date=current_date,
                end_date=current_date + timedelta(days=10)
            )
            if len(schedule) > 0:
                # Return first trading day after current date
                next_day = schedule.index[0].date()
                if next_day == current_date.date():
                    # If current date is a trading day, get next one
                    if len(schedule) > 1:
                        return schedule.index[1].date()
                return next_day
        except Exception as e:
            pass  # Fall back to simple logic
    
    # Fallback: simple weekend skip (doesn't handle holidays)
    next_day = current_date + timedelta(days=1)
    while next_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
        next_day += timedelta(days=1)
    return next_day.date()

def get_est_time():
    """Get current time in US Eastern timezone"""
    return datetime.now(pytz.timezone('US/Eastern'))

def is_sync_window():
    """Check if current time is within sync windows (7-8am or 7-8pm EST)"""
    now_est = get_est_time()
    return (7 <= now_est.hour < 8) or (19 <= now_est.hour < 20)

st.set_page_config(page_title="P2-ETF-Predictor", layout="wide")

# ------------------------------
# 2. POSITIONAL ENCODING LAYER
# ------------------------------
class PositionalEncoding(tf.keras.layers.Layer):
    """Adds positional information to input sequences for Transformer"""
    
    def __init__(self, max_seq_len=100, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_seq_len = max_seq_len
    
    def build(self, input_shape):
        """Build the layer - create positional encoding matrix"""
        seq_len = input_shape[1]
        d_model = input_shape[2]
        
        # Create position indices
        position = np.arange(0, seq_len, dtype=np.float32).reshape(-1, 1)
        
        # Create dimension indices - handle both even and odd d_model
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * 
                         -(np.log(10000.0) / d_model))
        
        # Calculate positional encodings
        pos_encoding = np.zeros((seq_len, d_model), dtype=np.float32)
        
        pos_encoding[:, 0::2] = np.sin(position * div_term)[:, :len(range(0, d_model, 2))]
        
        if d_model > 1:
            cos_values = np.cos(position * div_term)
            odd_positions = range(1, d_model, 2)
            pos_encoding[:, 1::2] = cos_values[:, :len(odd_positions)]
        
        self.pos_encoding = self.add_weight(
            name='positional_encoding',
            shape=(1, seq_len, d_model),
            initializer=tf.keras.initializers.Constant(pos_encoding),
            trainable=False
        )
        
        super(PositionalEncoding, self).build(input_shape)
    
    def call(self, inputs):
        return inputs + self.pos_encoding
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({'max_seq_len': self.max_seq_len})
        return config

# ------------------------------
# 3. DATA FETCHING FUNCTIONS
# ------------------------------
def fetch_macro_data_robust(start_date="2008-01-01"):
    """Fetch macro signals from multiple sources with proper error handling"""
    all_data = []
    data_sources = {}
    
    # 1. FRED Data
    try:
        fred_symbols = {
            "T10Y2Y": "T10Y2Y",
            "T10Y3M": "T10Y3M",
            "BAMLH0A0HYM2": "HY_Spread",
            "VIXCLS": "VIX",
            "DTWEXBGS": "DXY"
        }
        
        fred_data = web.DataReader(
            list(fred_symbols.keys()), 
            "fred", 
            start_date, 
            datetime.now()
        )
        fred_data.columns = [fred_symbols[col] for col in fred_data.columns]
        
        if fred_data.index.tz is not None:
            fred_data.index = fred_data.index.tz_localize(None)
        
        all_data.append(fred_data)
        data_sources['FRED'] = list(fred_data.columns)
        
    except Exception as e:
        st.warning(f"⚠️ FRED partial failure: {e}")
    
    # 2. Yahoo Finance Data
    try:
        yf_symbols = {
            "GC=F": "GOLD",
            "HG=F": "COPPER",
            "^VIX": "VIX_YF",
        }
        
        yf_data = yf.download(
            list(yf_symbols.keys()), 
            start=start_date, 
            progress=False,
            auto_adjust=True
        )['Close']
        
        if isinstance(yf_data, pd.Series):
            yf_data = yf_data.to_frame()
        
        yf_data.columns = [yf_symbols.get(col, col) for col in yf_data.columns]
        
        if yf_data.index.tz is not None:
            yf_data.index = yf_data.index.tz_localize(None)
        
        all_data.append(yf_data)
        data_sources['Yahoo'] = list(yf_data.columns)
        
    except Exception as e:
        st.warning(f"⚠️ Yahoo Finance failed: {e}")
    
    # 3. VIX Term Structure
    try:
        vix_term = yf.download(
            ["^VIX", "^VIX3M"],
            start=start_date,
            progress=False,
            auto_adjust=True
        )['Close']
        
        if not vix_term.empty:
            if isinstance(vix_term, pd.Series):
                vix_term = vix_term.to_frame()
            
            vix_term.columns = ["VIX_Spot", "VIX_3M"]
            vix_term['VIX_Term_Slope'] = vix_term['VIX_3M'] - vix_term['VIX_Spot']
            
            if vix_term.index.tz is not None:
                vix_term.index = vix_term.index.tz_localize(None)
            
            all_data.append(vix_term)
            data_sources['VIX_Term'] = list(vix_term.columns)
    
    except Exception as e:
        st.warning(f"⚠️ VIX Term Structure failed: {e}")
    
    # Combine all data sources
    if all_data:
        combined = pd.concat(all_data, axis=1, join='outer')
        combined = combined.loc[:, ~combined.columns.duplicated()]
        combined = combined.fillna(method='ffill', limit=5)
        return combined
    else:
        st.error("❌ Failed to fetch any macro data!")
        return pd.DataFrame()

def fetch_etf_data(etfs, start_date="2008-01-01"):
    """Fetch ETF price data and calculate returns"""
    
    try:
        etf_data = yf.download(
            etfs,
            start=start_date,
            progress=False,
            auto_adjust=True
        )['Close']
        
        if isinstance(etf_data, pd.Series):
            etf_data = etf_data.to_frame()
        
        if etf_data.index.tz is not None:
            etf_data.index = etf_data.index.tz_localize(None)
        
        # Calculate daily returns
        etf_returns = etf_data.pct_change()
        etf_returns.columns = [f"{col}_Ret" for col in etf_returns.columns]
        
        # Calculate 20-day realized volatility
        etf_vol = etf_data.pct_change().rolling(20).std() * np.sqrt(252)
        etf_vol.columns = [f"{col}_Vol" for col in etf_vol.columns]
        
        result = pd.concat([etf_returns, etf_vol], axis=1)
        
        return result
        
    except Exception as e:
        st.error(f"❌ ETF fetch failed: {e}")
        return pd.DataFrame()

# ------------------------------
# 4. HF DATASET SMART UPDATE
# ------------------------------
def smart_update_hf_dataset(new_data, token):
    """Smart update: Only uploads if new data exists or gaps are filled"""
    if not token:
        st.warning("⚠️ No HF_TOKEN found. Skipping dataset update.")
        return new_data
    
    raw_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/etf_data.csv"
    
    try:
        existing_df = pd.read_csv(raw_url)
        existing_df.columns = existing_df.columns.str.strip()
        
        date_col = next((c for c in existing_df.columns 
                        if c.lower() in ['date', 'unnamed: 0']), existing_df.columns[0])
        
        existing_df[date_col] = pd.to_datetime(existing_df[date_col])
        existing_df = existing_df.set_index(date_col).sort_index()
        
        if existing_df.index.tz is not None:
            existing_df.index = existing_df.index.tz_localize(None)
        
        combined = new_data.combine_first(existing_df)
        
        new_rows = len(combined) - len(existing_df)
        old_nulls = existing_df.isna().sum().sum()
        new_nulls = combined.isna().sum().sum()
        filled_gaps = old_nulls - new_nulls
        
        needs_update = new_rows > 0 or filled_gaps > 0
        
        if needs_update:
            combined.index.name = "Date"
            combined.reset_index().to_csv("etf_data.csv", index=False)
            
            api = HfApi()
            api.upload_file(
                path_or_fileobj="etf_data.csv",
                path_in_repo="etf_data.csv",
                repo_id=REPO_ID,
                repo_type="dataset",
                token=token,
                commit_message=f"Update: {get_est_time().strftime('%Y-%m-%d %H:%M EST')} | +{new_rows} rows, filled {filled_gaps} gaps"
            )
            
            st.success(f"✅ Dataset updated: +{new_rows} rows, filled {filled_gaps} gaps")
            
            return combined
        else:
            st.info("📊 Dataset already up-to-date. No upload needed.")
            return existing_df
            
    except Exception as e:
        st.warning(f"⚠️ Dataset update failed: {e}. Using new data only.")
        return new_data

# ------------------------------
# 5. MAIN DATA ENGINE
# ------------------------------
def get_data(start_year, force_refresh=False, clean_hf_dataset=False):
    """Main data fetching and processing pipeline"""
    raw_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/etf_data.csv"
    df = pd.DataFrame()
    
    try:
        df = pd.read_csv(raw_url)
        df.columns = df.columns.str.strip()
        
        date_col = next((c for c in df.columns if c.lower() in ['date', 'unnamed: 0']), df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        if clean_hf_dataset:
            st.warning("🧹 **Cleaning HF Dataset Mode Active**")
            original_cols = len(df.columns)
            
            nan_pct = (df.isna().sum() / len(df)) * 100
            bad_cols = nan_pct[nan_pct > 30].index.tolist()
            
            if bad_cols:
                st.write(f"📋 Found {len(bad_cols)} columns with >30% NaNs:")
                for col in bad_cols[:10]:
                    st.write(f"  - {col}: {nan_pct[col]:.1f}% NaNs")
                
                df = df.drop(columns=bad_cols)
                st.success(f"✅ Dropped {len(bad_cols)} columns ({original_cols} → {len(df.columns)})")
                
                token = os.getenv("HF_TOKEN")
                if token:
                    st.info("📤 Uploading cleaned dataset to HuggingFace...")
                    df.index.name = "Date"
                    df.reset_index().to_csv("etf_data.csv", index=False)
                    
                    api = HfApi()
                    api.upload_file(
                        path_or_fileobj="etf_data.csv",
                        path_in_repo="etf_data.csv",
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        token=token,
                        commit_message=f"Cleaned dataset: Removed {len(bad_cols)} columns with >30% NaNs"
                    )
                    st.success("✅ HF dataset updated!")
                else:
                    st.error("❌ HF_TOKEN not found. Cannot update dataset.")
            else:
                st.info("ℹ️ No columns found with >30% NaNs. Dataset is already clean.")
        
    except Exception as e:
        st.warning(f"⚠️ Could not load from HuggingFace: {e}")
    
    should_sync = is_sync_window() or force_refresh
    
    if should_sync:
        sync_reason = "🔄 Manual Refresh Requested" if force_refresh else "🔄 Sync Window Active"
        
        with st.status(f"{sync_reason} - Updating Dataset...", expanded=False):
            
            etf_list = ["TLT", "TBT", "VNQ", "SLV", "GLD", "AGG", "SPY"]
            
            etf_data = fetch_etf_data(etf_list, start_date="2008-01-01")
            macro_data = fetch_macro_data_robust(start_date="2008-01-01")
            
            if not etf_data.empty and not macro_data.empty:
                new_df = pd.concat([etf_data, macro_data], axis=1)
                
                token = os.getenv("HF_TOKEN")
                df = smart_update_hf_dataset(new_df, token)
            else:
                st.error("❌ Data fetch failed during sync")
    
    if df.empty:
        st.warning("📊 Fetching fresh data (no cached dataset available)...")
        etf_list = ["TLT", "TBT", "VNQ", "SLV", "GLD", "AGG", "SPY"]
        etf_data = fetch_etf_data(etf_list, start_date="2008-01-01")
        macro_data = fetch_macro_data_robust(start_date="2008-01-01")
        
        if not etf_data.empty and not macro_data.empty:
            df = pd.concat([etf_data, macro_data], axis=1)
    
    # Feature Engineering
    macro_cols = ['VIX', 'DXY', 'COPPER', 'GOLD', 'HY_Spread', 'T10Y2Y', 'T10Y3M', 
                  'VIX_Spot', 'VIX_3M', 'VIX_Term_Slope']
    
    nan_tracking = {}
    
    for col in df.columns:
        if any(m in col for m in macro_cols) or '_Vol' in col:
            rolling_mean = df[col].rolling(20, min_periods=5).mean()
            rolling_std = df[col].rolling(20, min_periods=5).std()
            z_col = f"{col}_Z"
            df[z_col] = (df[col] - rolling_mean) / (rolling_std + 1e-9)
            
            nan_count = df[z_col].isna().sum()
            if nan_count > 0:
                nan_tracking[z_col] = nan_count
    
    if nan_tracking:
        st.warning(f"⚠️ Features with NaNs: {len(nan_tracking)} features")
        worst_features = sorted(nan_tracking.items(), key=lambda x: x[1], reverse=True)[:5]
        for feat, count in worst_features:
            st.write(f"  - {feat}: {count} NaNs ({count/len(df)*100:.1f}%)")
    
    st.write("🎯 **Adding Regime Detection Features...**")
    
    st.info(f"📊 Raw data loaded: {len(df)} samples from {df.index[0].year} to {df.index[-1].year}")
    
    # VIX Regime
    if 'VIX' in df.columns:
        df['VIX_Regime_Low'] = (df['VIX'] < 15).astype(int)
        df['VIX_Regime_Med'] = ((df['VIX'] >= 15) & (df['VIX'] < 25)).astype(int)
        df['VIX_Regime_High'] = (df['VIX'] >= 25).astype(int)
    
    # Yield Curve Regime
    if 'T10Y2Y' in df.columns:
        df['YC_Inverted'] = (df['T10Y2Y'] < 0).astype(int)
        df['YC_Flat'] = ((df['T10Y2Y'] >= 0) & (df['T10Y2Y'] < 0.5)).astype(int)
        df['YC_Steep'] = (df['T10Y2Y'] >= 0.5).astype(int)
    
    # Credit Stress Regime
    if 'HY_Spread' in df.columns:
        df['Credit_Stress_Low'] = (df['HY_Spread'] < 400).astype(int)
        df['Credit_Stress_Med'] = ((df['HY_Spread'] >= 400) & (df['HY_Spread'] < 600)).astype(int)
        df['Credit_Stress_High'] = (df['HY_Spread'] >= 600).astype(int)
    
    # VIX Term Structure Regime
    if 'VIX_Term_Slope' in df.columns:
        df['VIX_Term_Contango'] = (df['VIX_Term_Slope'] > 2).astype(int)
        df['VIX_Term_Backwardation'] = (df['VIX_Term_Slope'] < -2).astype(int)
    
    # Rate Environment
    if 'T10Y3M' in df.columns:
        df['Rates_VeryLow'] = (df['T10Y3M'] < 1.0).astype(int)
        df['Rates_Low'] = ((df['T10Y3M'] >= 1.0) & (df['T10Y3M'] < 2.0)).astype(int)
        df['Rates_Normal'] = ((df['T10Y3M'] >= 2.0) & (df['T10Y3M'] < 3.0)).astype(int)
        df['Rates_High'] = (df['T10Y3M'] >= 3.0).astype(int)
    
    st.success(f"✅ Added {len([c for c in df.columns if 'Regime' in c or 'YC_' in c or 'Credit_' in c or 'Rates_' in c])} regime features")
    
    df = df[df.index.year >= start_year]
    st.info(f"📅 After year filter ({start_year}+): {len(df)} samples from {df.index[0].year if len(df) > 0 else 'N/A'} to {df.index[-1].year if len(df) > 0 else 'N/A'}")
    
    # Cleaning
    nan_percentages = df.isna().sum() / len(df)
    bad_features = nan_percentages[nan_percentages > 0.5].index.tolist()
    
    if bad_features:
        st.warning(f"🗑️ Dropping {len(bad_features)} features with >50% NaNs: {bad_features[:5]}...")
        df = df.drop(columns=bad_features)
    
    df = df.fillna(method='ffill', limit=5)
    df = df.fillna(method='bfill', limit=100)
    df = df.fillna(method='ffill')
    
    nan_count_before = df.isna().sum().sum()
    df = df.dropna()
    
    if len(df) > 0:
        st.success(f"✅ Final dataset: {len(df)} samples from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')} (dropped {nan_count_before} remaining NaN cells)")
    else:
        st.error("❌ No data remaining after cleaning!")
    
    return df

# ------------------------------
# 6. CUSTOM LOSS FUNCTION
# ------------------------------
def directional_loss(y_true, y_pred):
    """Custom loss that penalizes incorrect direction predictions more heavily"""
    abs_error = tf.abs(y_true - y_pred)
    signs_match = tf.cast(tf.math.sign(y_true) == tf.math.sign(y_pred), tf.float32)
    penalty = tf.where(signs_match > 0.5, abs_error, abs_error * 2.0)
    return tf.reduce_mean(penalty)

# ------------------------------
# 7. MODEL BUILDERS
# ------------------------------
def build_pure_transformer(input_shape, num_outputs, num_heads=2, ff_dim=64, num_layers=1, dropout_rate=0.2):
    """Build a pure Transformer architecture"""
    inputs = Input(shape=input_shape)
    x = PositionalEncoding()(inputs)
    
    for _ in range(num_layers):
        attn_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=input_shape[1] // num_heads,
            dropout=dropout_rate
        )(x, x)
        
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        ff_output = Dense(ff_dim, activation='relu', kernel_regularizer=l2(0.01))(x)
        ff_output = Dropout(dropout_rate)(ff_output)
        ff_output = Dense(input_shape[1], kernel_regularizer=l2(0.01))(ff_output)
        
        x = LayerNormalization(epsilon=1e-6)(x + ff_output)
    
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(ff_dim, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_outputs)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# ------------------------------
# 8. SIDEBAR CONFIGURATION
# ------------------------------
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    current_time = get_est_time()
    st.write(f"🕒 **Server Time (EST):** {current_time.strftime('%H:%M:%S')}")
    
    if is_sync_window():
        st.success("✅ **Sync Window Active**")
    else:
        st.info("⏸️ Sync Window Inactive")
    
    st.divider()
    
    st.subheader("📥 Dataset Management")
    force_refresh = st.checkbox(
        "Force Dataset Refresh",
        value=False,
        help="Manually fetch fresh data and update HF dataset (ignores sync window)"
    )
    
    clean_dataset = st.checkbox(
        "Clean HF Dataset (Remove NaN-heavy columns)",
        value=False,
        help="Remove columns with >30% missing data from HF dataset permanently"
    )
    
    st.divider()
    
    start_yr = st.slider("📅 Start Year", 2008, 2025, 2016)
    fee_bps = st.slider("💰 Transaction Fee (bps)", 0, 100, 15, 
                        help="Transaction cost in basis points")
    
    st.divider()
    
    st.subheader("🧠 Model Selection")
    model_option = st.selectbox(
        "Choose Model",
        [
            "Option A: Pure Transformer",
            "Option B: Random Forest + XGBoost"
        ],
        index=1
    )
    
    st.divider()
    
    st.subheader("⚙️ Training Settings")
    
    if "Option A" in model_option:
        epochs = st.number_input("Epochs", 10, 500, 100, step=10)
        lookback = st.slider("Lookback Days", 20, 60, 30, step=5)
        st.caption("ℹ️ Transformer: 2 heads, 1 layer, 64 FF dim")
    else:
        epochs = 100
        lookback = 30
        st.info("ℹ️ RF+XGBoost uses optimized settings: 500 trees/rounds with early stopping")
    
    st.divider()
    
    st.subheader("📊 Data Split Strategy")
    
    split_option = st.selectbox(
        "Train/Val/Test Split",
        ["70/15/15", "80/10/10"],
        index=0,
        help="Choose data split ratio: Train/Validation/Out-of-Sample"
    )
    
    split_ratios = {
        "70/15/15": (0.70, 0.15, 0.15),
        "80/10/10": (0.80, 0.10, 0.10)
    }
    train_pct, val_pct, test_pct = split_ratios[split_option]
    
    st.divider()
    
    run_button = st.button("🚀 Execute Model", type="primary", use_container_width=True)
    
    st.divider()
    
    refresh_only_button = st.button("🔄 Refresh Dataset Only", type="secondary", use_container_width=True, 
                                    help="Update HF dataset with latest data without training the model")

# ------------------------------
# 9. MAIN APPLICATION
# ------------------------------
st.title("🤖 P2-ETF-PREDICTOR")
st.caption("Multi-Model Ensemble: Transformer, Random Forest, XGBoost")

if refresh_only_button:
    st.info("🔄 Refreshing dataset without model training...")
    
    with st.status("📡 Fetching fresh data from all sources...", expanded=True):
        etf_list = ["TLT", "TBT", "VNQ", "SLV", "GLD", "AGG", "SPY"]
        
        etf_data = fetch_etf_data(etf_list, start_date="2008-01-01")
        macro_data = fetch_macro_data_robust(start_date="2008-01-01")
        
        if not etf_data.empty and not macro_data.empty:
            new_df = pd.concat([etf_data, macro_data], axis=1)
            
            token = os.getenv("HF_TOKEN")
            
            if token:
                updated_df = smart_update_hf_dataset(new_df, token)
                
                st.success("✅ Dataset refresh completed!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Rows", len(updated_df))
                col2.metric("Total Columns", len(updated_df.columns))
                col3.metric("Date Range", f"{updated_df.index[0].strftime('%Y-%m-%d')} to {updated_df.index[-1].strftime('%Y-%m-%d')}")
                
            else:
                st.error("❌ HF_TOKEN not found. Cannot update dataset.")
        else:
            st.error("❌ Failed to fetch data from sources")
    
    st.stop()

if run_button:
    df = get_data(start_yr, force_refresh=force_refresh, clean_hf_dataset=clean_dataset)
    
    if df.empty:
        st.error("❌ No data available. Please check data sources.")
        st.stop()
    
    years_of_data = df.index[-1].year - df.index[0].year + 1
    
    st.write(f"📅 **Data Range:** {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')} ({years_of_data} years)")
    
    validation_errors = []
    validation_warnings = []
    
    if split_option == "80/10/10" and years_of_data < 10:
        validation_warnings.append(f"""
        ⚠️ **80/10/10 with {years_of_data} years: UNSTABLE**
        - Needs 12+ years for reliability (you have {years_of_data})
        - Test set will be very small (< 6 months)
        - May work but results will be high variance
        """)
    
    train_pct, val_pct, test_pct = split_ratios[split_option]
    estimated_test_samples = int(len(df) * test_pct)
    
    if estimated_test_samples < 200:
        validation_warnings.append(f"""
        ⚠️ **Test Set Too Small: {estimated_test_samples} samples**
        - Minimum recommended: 250 samples (1 year)
        - Small test sets produce unreliable metrics
        - Consider using earlier start year or different split
        """)
    
    if validation_errors:
        for error in validation_errors:
            st.error(error)
        st.error("**❌ STOP: Configuration will likely fail. Please adjust settings above.**")
        if st.button("⚠️ Run Anyway (Not Recommended)", type="secondary"):
            st.warning("Proceeding despite warnings...")
        else:
            st.stop()
    
    if validation_warnings:
        for warning in validation_warnings:
            st.warning(warning)
        st.info("💡 **Recommendation:** Use 70/15/15 split OR choose earlier start year (2008-2012)")
    
    if not validation_errors and not validation_warnings:
        st.success(f"✅ Configuration validated: {years_of_data} years, {estimated_test_samples} test samples")
    
    target_etfs = [col for col in df.columns if col.endswith('_Ret') and 
                   any(etf in col for etf in ['TLT', 'TBT', 'VNQ', 'SLV', 'GLD'])]
    
    benchmark_etfs = ['AGG_Ret', 'SPY_Ret']
    
    if not target_etfs:
        st.error("❌ No target ETF returns found in dataset")
        st.stop()
    
    input_features = [col for col in df.columns 
                     if (col.endswith('_Z') or col.endswith('_Vol') or 
                         'Regime' in col or 'YC_' in col or 'Credit_' in col or 
                         'Rates_' in col or 'VIX_Term_' in col)
                     and col not in target_etfs]
    
    if not input_features:
        st.error("❌ No input features found in dataset")
        st.stop()
    
    st.info(f"🎯 **Targets:** {len(target_etfs)} ETFs | **Features:** {len(input_features)} signals | **Model:** {model_option}")
    
    
    # Prepare data based on model type
    if "Option B" in model_option:
        X = df[input_features].values
        y_raw = df[target_etfs].values
        
        y = np.argmax(y_raw, axis=1)
        
        train_size = int(len(X) * train_pct)
        val_size = int(len(X) * val_pct)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        y_raw_train = y_raw[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        y_raw_val = y_raw[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        y_raw_test = y_raw[train_size + val_size:]
        
        st.success(f"✅ Split ({split_option}): Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
        
        with st.spinner("🌲 Training Random Forest + XGBoost Ensemble..."):
            rf_model = RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=3,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=50,
                eval_metric='mlogloss'
            )
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            st.success("✅ Ensemble training completed!")
        
        rf_probs = rf_model.predict_proba(X_test)
        xgb_probs = xgb_model.predict_proba(X_test)
        
        ensemble_probs = (rf_probs + xgb_probs) / 2
        preds = np.argmax(ensemble_probs, axis=1)
        
    else:
        scaler = MinMaxScaler()
        scaled_input = scaler.fit_transform(df[input_features])
        
        X, y = [], []
        for i in range(lookback, len(scaled_input)):
            X.append(scaled_input[i-lookback:i])
            y.append(df[target_etfs].iloc[i].values)
        
        X = np.array(X)
        y = np.array(y)
        
        train_size = int(len(X) * train_pct)
        val_size = int(len(X) * val_pct)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        st.success(f"✅ Split ({split_option}): Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
        
        if "Option A" in model_option:
            with st.spinner("🧠 Training Pure Transformer Model..."):
                model = build_pure_transformer(
                    input_shape=(X.shape[1], X.shape[2]),
                    num_outputs=len(target_etfs),
                    num_heads=2,
                    ff_dim=64,
                    num_layers=1,
                    dropout_rate=0.2
                )
                
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss=directional_loss,
                    metrics=['mae']
                )
                
                early_stop = EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True,
                    verbose=0
                )
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=int(epochs),
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=0
                )
                
                st.success(f"✅ Training completed! Best epoch: {len(history.history['loss']) - 20}")
            
            preds = model.predict(X_test, verbose=0)
    
    # ============================================================
    # CORRECTED: Get test dates and filter to NYSE trading days
    # ============================================================
    if "Option B" in model_option:
        test_dates_all = df.index[train_size + val_size:]
    else:
        test_dates_all = df.index[lookback + train_size + val_size:]
    
    # ✅ CRITICAL: Filter to only NYSE trading days
    if NYSE_CALENDAR_AVAILABLE:
        try:
            nyse = mcal.get_calendar('NYSE')
            trading_schedule = nyse.schedule(
                start_date=test_dates_all[0].strftime('%Y-%m-%d'),
                end_date=test_dates_all[-1].strftime('%Y-%m-%d')
            )
            valid_trading_days = trading_schedule.index.normalize()
            
            if valid_trading_days.tz is not None:
                valid_trading_days = valid_trading_days.tz_localize(None)
            
            test_dates = test_dates_all[test_dates_all.isin(valid_trading_days)]
            st.success(f"✅ Filtered to {len(test_dates)} NYSE trading days (removed {len(test_dates_all) - len(test_dates)} weekend/holiday days)")
        except Exception as e:
            st.warning(f"⚠️ NYSE calendar filter failed: {e}. Using all dates.")
            test_dates = test_dates_all
    else:
        test_dates = test_dates_all
    
    sofr = df['T10Y3M'].iloc[-1] / 100 if 'T10Y3M' in df.columns else 0.045
    
    # ============================================================
    # STRATEGY EXECUTION
    # ============================================================
    strat_rets = []
    audit_trail = []
    
    for i in range(len(preds)):
        if "Option B" in model_option:
            best_idx = preds[i]
            signal_etf = target_etfs[best_idx].replace('_Ret', '')
            realized_ret = y_raw_test[i][best_idx]
            
        else:
            best_idx = np.argmax(preds[i])
            signal_etf = target_etfs[best_idx].replace('_Ret', '')
            realized_ret = y_test[i][best_idx]
        
        net_ret = realized_ret - (fee_bps / 10000)
        
        strat_rets.append(net_ret)
        
        audit_trail.append({
            'Date': test_dates[i].strftime('%Y-%m-%d'),
            'Signal': signal_etf,
            'Realized': realized_ret,
            'Net_Return': net_ret
        })
    
    strat_rets = np.array(strat_rets)
    
    # Get next trading day
    if len(test_dates) > 0:
        last_date = test_dates[-1]
        next_trading = get_next_trading_day(last_date)
        
        if "Option B" in model_option:
            if len(preds) > 0:
                next_best_idx = preds[-1]
                next_signal = target_etfs[next_best_idx].replace('_Ret', '')
            else:
                next_signal = "CASH"
        else:
            if len(preds) > 0:
                next_best_idx = np.argmax(preds[-1])
                next_signal = target_etfs[next_best_idx].replace('_Ret', '')
            else:
                next_signal = "CASH"
    else:
        next_trading = datetime.now().date()
        next_signal = "CASH"
    
    # Performance Analytics
    st.divider()
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #00d1b2 0%, #00a896 100%); 
                padding: 25px; 
                border-radius: 15px; 
                text-align: center;
                box-shadow: 0 8px 16px rgba(0,0,0,0.3);
                margin: 20px 0;">
        <h1 style="color: white; 
                   font-size: 48px; 
                   margin: 0 0 10px 0;
                   font-weight: bold;
                   text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🎯 NEXT TRADING DAY
        </h1>
        <h2 style="color: white; 
                   font-size: 36px; 
                   margin: 0;
                   font-weight: bold;">
            {next_trading.strftime('%Y-%m-%d')} → {next_signal}
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Calculate metrics
    cum_returns = np.cumprod(1 + strat_rets)
    ann_return = (cum_returns[-1] ** (252 / len(strat_rets))) - 1
    sharpe = (np.mean(strat_rets) - (sofr / 252)) / (np.std(strat_rets) + 1e-9) * np.sqrt(252)
    
    recent_rets = strat_rets[-15:]
    hit_ratio = np.mean(recent_rets > 0)
    
    cum_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - cum_max) / cum_max
    max_dd = np.min(drawdown)
    
    max_daily_dd = np.min(strat_rets)
    
    # Calculate benchmark returns
    agg_returns = None
    spy_returns = None
    
    if 'AGG_Ret' in df.columns:
        if "Option B" in model_option:
            agg_returns = df['AGG_Ret'].iloc[train_size + val_size:].values[:len(strat_rets)]
        else:
            agg_returns = df['AGG_Ret'].iloc[lookback + train_size + val_size:].values[:len(strat_rets)]
        agg_cum_returns = np.cumprod(1 + agg_returns)
    
    if 'SPY_Ret' in df.columns:
        if "Option B" in model_option:
            spy_returns = df['SPY_Ret'].iloc[train_size + val_size:].values[:len(strat_rets)]
        else:
            spy_returns = df['SPY_Ret'].iloc[lookback + train_size + val_size:].values[:len(strat_rets)]
        spy_cum_returns = np.cumprod(1 + spy_returns)
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric(
        "📈 Annualized Return",
        f"{ann_return * 100:.2f}%",
        delta=f"vs SOFR: {(ann_return - sofr) * 100:.2f}%"
    )
    
    col2.metric(
        "📊 Sharpe Ratio",
        f"{sharpe:.2f}",
        delta="Risk-Adjusted" if sharpe > 1 else "Below Threshold"
    )
    
    col3.metric(
        "🎯 Hit Ratio (15d)",
        f"{hit_ratio * 100:.0f}%",
        delta="Strong" if hit_ratio > 0.6 else "Weak"
    )
    
    col4.metric(
        "📉 Max Drawdown",
        f"{max_dd * 100:.2f}%",
        delta="Peak to Trough",
        help="Maximum decline from peak to trough during OOS period"
    )
    
    col5.metric(
        "⚠️ Max Daily DD",
        f"{max_daily_dd * 100:.2f}%",
        delta="Worst Day",
        help="Largest single-day loss during OOS period"
    )
    
    # Equity curve
    st.subheader("📈 Out-of-Sample Equity Curve (with Benchmarks)")
    
    fig_equity = go.Figure()
    
    fig_equity.add_trace(go.Scatter(
        x=test_dates[:len(cum_returns)],
        y=cum_returns,
        mode='lines',
        name='Strategy',
        line=dict(color='#00d1b2', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 209, 178, 0.1)'
    ))
    
    fig_equity.add_trace(go.Scatter(
        x=test_dates[:len(cum_max)],
        y=cum_max,
        mode='lines',
        name='High Water Mark',
        line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dash')
    ))
    
    if agg_returns is not None:
        fig_equity.add_trace(go.Scatter(
            x=test_dates[:len(agg_cum_returns)],
            y=agg_cum_returns,
            mode='lines',
            name='AGG (Bond Benchmark)',
            line=dict(color='#ffa500', width=2, dash='dot')
        ))
    
    if spy_returns is not None:
        fig_equity.add_trace(go.Scatter(
            x=test_dates[:len(spy_cum_returns)],
            y=spy_cum_returns,
            mode='lines',
            name='SPY (Equity Benchmark)',
            line=dict(color='#ff4b4b', width=2, dash='dot')
        ))
    
    fig_equity.update_layout(
        template="plotly_dark",
        height=450,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        xaxis_title="Date",
        yaxis_title="Cumulative Return"
    )
    
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # Benchmark comparison
    if agg_returns is not None or spy_returns is not None:
        st.subheader("📊 Benchmark Comparison")
        
        comparison_data = {
            'Metric': ['Annualized Return', 'Sharpe Ratio', 'Max Drawdown', 'Max Daily DD'],
            'Strategy': [
                f"{ann_return * 100:.2f}%",
                f"{sharpe:.2f}",
                f"{max_dd * 100:.2f}%",
                f"{max_daily_dd * 100:.2f}%"
            ]
        }
        
        if agg_returns is not None:
            agg_ann_return = (agg_cum_returns[-1] ** (252 / len(agg_returns))) - 1
            agg_sharpe = (np.mean(agg_returns) - (sofr / 252)) / (np.std(agg_returns) + 1e-9) * np.sqrt(252)
            agg_cum_max = np.maximum.accumulate(agg_cum_returns)
            agg_dd = (agg_cum_returns - agg_cum_max) / agg_cum_max
            agg_max_dd = np.min(agg_dd)
            agg_max_daily_dd = np.min(agg_returns)
            
            comparison_data['AGG (Bonds)'] = [
                f"{agg_ann_return * 100:.2f}%",
                f"{agg_sharpe:.2f}",
                f"{agg_max_dd * 100:.2f}%",
                f"{agg_max_daily_dd * 100:.2f}%"
            ]
        
        if spy_returns is not None:
            spy_ann_return = (spy_cum_returns[-1] ** (252 / len(spy_returns))) - 1
            spy_sharpe = (np.mean(spy_returns) - (sofr / 252)) / (np.std(spy_returns) + 1e-9) * np.sqrt(252)
            spy_cum_max = np.maximum.accumulate(spy_cum_returns)
            spy_dd = (spy_cum_returns - spy_cum_max) / spy_cum_max
            spy_max_dd = np.min(spy_dd)
            spy_max_daily_dd = np.min(spy_returns)
            
            comparison_data['SPY (Equity)'] = [
                f"{spy_ann_return * 100:.2f}%",
                f"{spy_sharpe:.2f}",
                f"{spy_max_dd * 100:.2f}%",
                f"{spy_max_daily_dd * 100:.2f}%"
            ]
        
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df)
    
    # Training history
    if "Option A" in model_option:
        st.subheader("📉 Training & Validation Loss")
        
        fig_loss = make_subplots(specs=[[{"secondary_y": False}]])
        
        fig_loss.add_trace(go.Scatter(
            x=list(range(len(history.history['loss']))),
            y=history.history['loss'],
            mode='lines',
            name='Training Loss',
            line=dict(color='#00d1b2', width=2)
        ))
        
        fig_loss.add_trace(go.Scatter(
            x=list(range(len(history.history['val_loss']))),
            y=history.history['val_loss'],
            mode='lines',
            name='Validation Loss',
            line=dict(color='#ff4b4b', width=2)
        ))
        
        fig_loss.update_layout(
            template="plotly_dark",
            height=300,
            xaxis_title="Epoch",
            yaxis_title="Loss",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_loss, use_container_width=True)
    
    # Audit trail
    st.subheader("📋 Last 15 Days Audit Trail")
    
    audit_df = pd.DataFrame(audit_trail).tail(15)[['Date', 'Signal', 'Net_Return']]
    
    def color_return(val):
        return 'color: #00ff00; font-weight: bold' if val > 0 else 'color: #ff4b4b; font-weight: bold'
    
    styled_audit = audit_df.style.applymap(
        color_return,
        subset=['Net_Return']
    ).format({
        'Net_Return': '{:.2%}'
    }).set_properties(**{
        'font-size': '18px',
        'text-align': 'center'
    }).set_table_styles([
        {'selector': 'th', 'props': [('font-size', '20px'), ('font-weight', 'bold'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('padding', '12px')]}
    ])
    
    st.dataframe(styled_audit, use_container_width=True, height=600)
    
    # Model summary
    with st.expander("🔧 Model Architecture Details"):
        st.text(f"Selected Model: {model_option}")
        
        if "Option A" in model_option:
            st.text("Pure Transformer Configuration:")
            st.text(f"  - Input Shape: ({lookback}, {len(input_features)})")
            st.text(f"  - Attention Heads: 2")
            st.text(f"  - Transformer Layers: 1")
            st.text(f"  - Feed-Forward Dim: 64")
            st.text(f"  - Dropout Rate: 0.2")
            st.text(f"  - L2 Regularization: 0.01")
            st.text(f"  - Loss Function: Directional Loss")
            st.text(f"  - Total Parameters: {model.count_params():,}")
        
        elif "Option B" in model_option:
            st.text("Random Forest + XGBoost Ensemble (Enhanced):")
            st.text(f"  - Random Forest: 500 trees, max_depth=15, min_leaf=3")
            st.text(f"  - XGBoost: 500 rounds, max_depth=8, lr=0.03")
            st.text(f"  - XGBoost Regularization: L1=0.1, L2=1.0, gamma=0.1")
            st.text(f"  - Early Stopping: 50 rounds")
            st.text(f"  - Input Features: {len(input_features)}")
            st.text(f"  - Ensemble: Average probabilities")
        
        st.text(f"\nData Split: {split_option}")
        st.text(f"  - Training Samples: {len(X_train) if 'X_train' in locals() else 'N/A'}")
        st.text(f"  - Validation Samples: {len(X_val) if 'X_val' in locals() else 'N/A'}")
        st.text(f"  - Test Samples: {len(X_test) if 'X_test' in locals() else 'N/A'}")

else:
    st.info("👈 Configure parameters in the sidebar and click '🚀 Execute Model' to begin")
    
    current_time = get_est_time()
    st.write(f"🕒 Current EST Time: **{current_time.strftime('%H:%M:%S')}**")
    
    if is_sync_window():
        st.success("✅ **Sync Window Active** - Data will be updated from sources")
    else:
        next_sync = "07:00-08:00" if current_time.hour < 7 else "19:00-20:00"
        st.info(f"⏸️ Next sync window: **{next_sync} EST**")
