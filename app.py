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
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------
# 1. CORE CONFIG & SYNC WINDOW
# ------------------------------
REPO_ID = "P2SAMAPA/my-etf-data"

def get_est_time():
    """Get current time in US Eastern timezone"""
    return datetime.now(pytz.timezone('US/Eastern'))

def is_sync_window():
    """Check if current time is within sync windows (7-8am or 7-8pm EST)"""
    now_est = get_est_time()
    return (7 <= now_est.hour < 8) or (19 <= now_est.hour < 20)

st.set_page_config(page_title="P2-Transformer Pro", layout="wide")

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
        
        # For even dimensions: d_model = 10 → indices 0,2,4,6,8 (5 values each)
        # For odd dimensions: d_model = 11 → indices 0,2,4,6,8,10 (sin gets 6, cos gets 5)
        pos_encoding[:, 0::2] = np.sin(position * div_term)[:, :len(range(0, d_model, 2))]
        
        # Cosine for odd indices - handle case where we might have fewer odd indices
        if d_model > 1:
            cos_values = np.cos(position * div_term)
            # Determine how many odd positions we have
            odd_positions = range(1, d_model, 2)
            pos_encoding[:, 1::2] = cos_values[:, :len(odd_positions)]
        
        # Store as a non-trainable weight
        self.pos_encoding = self.add_weight(
            name='positional_encoding',
            shape=(1, seq_len, d_model),
            initializer=tf.keras.initializers.Constant(pos_encoding),
            trainable=False
        )
        
        super(PositionalEncoding, self).build(input_shape)
    
    def call(self, inputs):
        """Add positional encoding to inputs"""
        return inputs + self.pos_encoding
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({'max_seq_len': self.max_seq_len})
        return config

# ------------------------------
# 3. ROBUST DATA FETCHING ENGINE
# ------------------------------
def fetch_macro_data_robust(start_date="2008-01-01"):
    """
    Fetch macro signals from multiple sources with proper error handling
    Returns: Combined DataFrame with all available macro signals
    """
    all_data = []
    data_sources = {}
    
    # 1. FRED Data (Most Reliable)
    st.write("📊 Fetching FRED data...")
    try:
        fred_symbols = {
            "T10Y2Y": "T10Y2Y",      # 10Y-2Y Treasury Spread
            "T10Y3M": "T10Y3M",      # 10Y-3M Treasury Spread  
            "BAMLH0A0HYM2": "HY_Spread",  # High Yield Credit Spread
            "VIXCLS": "VIX",         # VIX from FRED
            "DTWEXBGS": "DXY"        # Dollar Index from FRED
        }
        
        fred_data = web.DataReader(
            list(fred_symbols.keys()), 
            "fred", 
            start_date, 
            datetime.now()
        )
        fred_data.columns = [fred_symbols[col] for col in fred_data.columns]
        
        # Remove timezone if present
        if fred_data.index.tz is not None:
            fred_data.index = fred_data.index.tz_localize(None)
        
        all_data.append(fred_data)
        data_sources['FRED'] = list(fred_data.columns)
        st.success(f"✅ FRED: {len(fred_data.columns)} signals fetched")
        
    except Exception as e:
        st.warning(f"⚠️ FRED partial failure: {e}")
        # Try individual symbols
        for symbol, name in fred_symbols.items():
            try:
                temp = web.DataReader(symbol, "fred", start_date, datetime.now())
                temp.columns = [name]
                if temp.index.tz is not None:
                    temp.index = temp.index.tz_localize(None)
                all_data.append(temp)
            except:
                pass
    
    # 2. Yahoo Finance Data
    st.write("📊 Fetching Yahoo Finance data...")
    try:
        yf_symbols = {
            "GC=F": "GOLD",      # Gold Futures
            "HG=F": "COPPER",    # Copper Futures
            "^VIX": "VIX_YF",    # VIX (backup if FRED fails)
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
        
        # Remove timezone if present
        if yf_data.index.tz is not None:
            yf_data.index = yf_data.index.tz_localize(None)
        
        all_data.append(yf_data)
        data_sources['Yahoo'] = list(yf_data.columns)
        st.success(f"✅ Yahoo: {len(yf_data.columns)} signals fetched")
        
    except Exception as e:
        st.warning(f"⚠️ Yahoo Finance failed: {e}")
    
    # 3. VIX Term Structure (Standard Macro Signal)
    st.write("📊 Fetching VIX term structure...")
    try:
        vix_term = yf.download(
            ["^VIX", "^VIX3M"],  # VIX and 3-month VIX
            start=start_date,
            progress=False,
            auto_adjust=True
        )['Close']
        
        if not vix_term.empty:
            if isinstance(vix_term, pd.Series):
                vix_term = vix_term.to_frame()
            
            vix_term.columns = ["VIX_Spot", "VIX_3M"]
            
            # Calculate VIX term structure slope (measures volatility curve steepness)
            vix_term['VIX_Term_Slope'] = vix_term['VIX_3M'] - vix_term['VIX_Spot']
            
            # Remove timezone
            if vix_term.index.tz is not None:
                vix_term.index = vix_term.index.tz_localize(None)
            
            all_data.append(vix_term)
            data_sources['VIX_Term'] = list(vix_term.columns)
            st.success(f"✅ VIX Term Structure: {len(vix_term.columns)} signals")
    
    except Exception as e:
        st.warning(f"⚠️ VIX Term Structure failed: {e}")
    
    # Combine all data sources
    if all_data:
        combined = pd.concat(all_data, axis=1, join='outer')
        
        # Remove duplicate columns (keep first occurrence)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        
        # Forward fill missing values (max 5 days)
        combined = combined.fillna(method='ffill', limit=5)
        
        st.info(f"📈 Total macro signals: {len(combined.columns)} from {len(data_sources)} sources")
        
        return combined
    else:
        st.error("❌ Failed to fetch any macro data!")
        return pd.DataFrame()

def fetch_etf_data(etfs, start_date="2008-01-01"):
    """
    Fetch ETF price data and calculate returns
    """
    st.write(f"📊 Fetching ETF data for: {', '.join(etfs)}...")
    
    try:
        etf_data = yf.download(
            etfs,
            start=start_date,
            progress=False,
            auto_adjust=True
        )['Close']
        
        if isinstance(etf_data, pd.Series):
            etf_data = etf_data.to_frame()
        
        # Remove timezone
        if etf_data.index.tz is not None:
            etf_data.index = etf_data.index.tz_localize(None)
        
        # Calculate daily returns
        etf_returns = etf_data.pct_change()
        etf_returns.columns = [f"{col}_Ret" for col in etf_returns.columns]
        
        # Calculate 20-day realized volatility
        etf_vol = etf_data.pct_change().rolling(20).std() * np.sqrt(252)
        etf_vol.columns = [f"{col}_Vol" for col in etf_vol.columns]
        
        # Combine
        result = pd.concat([etf_returns, etf_vol], axis=1)
        
        st.success(f"✅ ETF data: {len(etf_data.columns)} ETFs, {len(result.columns)} features")
        
        return result
        
    except Exception as e:
        st.error(f"❌ ETF fetch failed: {e}")
        return pd.DataFrame()

# ------------------------------
# 4. HF DATASET SMART UPDATE
# ------------------------------
def smart_update_hf_dataset(new_data, token):
    """
    Smart update: Only uploads if new data exists or gaps are filled
    """
    if not token:
        st.warning("⚠️ No HF_TOKEN found. Skipping dataset update.")
        return new_data
    
    raw_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/etf_data.csv"
    
    try:
        # Download existing dataset
        existing_df = pd.read_csv(raw_url)
        existing_df.columns = existing_df.columns.str.strip()
        
        # Find date column
        date_col = next((c for c in existing_df.columns 
                        if c.lower() in ['date', 'unnamed: 0']), existing_df.columns[0])
        
        existing_df[date_col] = pd.to_datetime(existing_df[date_col])
        existing_df = existing_df.set_index(date_col).sort_index()
        
        # Remove timezone if present
        if existing_df.index.tz is not None:
            existing_df.index = existing_df.index.tz_localize(None)
        
        st.info(f"📥 Existing dataset: {len(existing_df)} rows, {len(existing_df.columns)} columns")
        
        # Merge: New data takes priority for overlapping dates
        combined = new_data.combine_first(existing_df)
        
        # Calculate improvements
        new_rows = len(combined) - len(existing_df)
        old_nulls = existing_df.isna().sum().sum()
        new_nulls = combined.isna().sum().sum()
        filled_gaps = old_nulls - new_nulls
        
        # Decide if upload is needed
        needs_update = new_rows > 0 or filled_gaps > 0
        
        if needs_update:
            # Prepare and upload
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
def get_data(start_year, force_refresh=False):
    """
    Main data fetching and processing pipeline
    
    Args:
        start_year: Filter data from this year onwards
        force_refresh: If True, fetch fresh data regardless of sync window
    """
    # Always try to load from HF first
    raw_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/etf_data.csv"
    df = pd.DataFrame()
    
    try:
        st.info("📥 Loading dataset from HuggingFace...")
        df = pd.read_csv(raw_url)
        df.columns = df.columns.str.strip()
        
        # Fix date column
        date_col = next((c for c in df.columns if c.lower() in ['date', 'unnamed: 0']), df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        
        # Remove timezone
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        st.success(f"✅ Loaded {len(df)} rows from HuggingFace")
        
    except Exception as e:
        st.warning(f"⚠️ Could not load from HuggingFace: {e}")
    
    # Check if we should sync (during sync windows OR force refresh)
    should_sync = is_sync_window() or force_refresh
    
    if should_sync:
        sync_reason = "🔄 Manual Refresh Requested" if force_refresh else "🔄 Sync Window Active"
        
        with st.status(f"{sync_reason} - Updating Dataset...", expanded=True):
            st.write(f"🕒 Current time (EST): {get_est_time().strftime('%H:%M:%S')}")
            
            if force_refresh:
                st.info("📡 Force refresh enabled - fetching data outside sync window")
            
            # Fetch fresh data
            etf_list = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
            
            # Fetch ETF data
            etf_data = fetch_etf_data(etf_list, start_date="2008-01-01")
            
            # Fetch macro data
            macro_data = fetch_macro_data_robust(start_date="2008-01-01")
            
            # Combine all
            if not etf_data.empty and not macro_data.empty:
                new_df = pd.concat([etf_data, macro_data], axis=1)
                
                # Update HF dataset
                token = os.getenv("HF_TOKEN")
                df = smart_update_hf_dataset(new_df, token)
                
                if force_refresh:
                    st.success("✅ Manual refresh completed successfully!")
            else:
                st.error("❌ Data fetch failed during sync")
    
    # If still empty, fetch fresh data anyway
    if df.empty:
        st.warning("📊 Fetching fresh data (no cached dataset available)...")
        etf_list = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
        etf_data = fetch_etf_data(etf_list, start_date="2008-01-01")
        macro_data = fetch_macro_data_robust(start_date="2008-01-01")
        
        if not etf_data.empty and not macro_data.empty:
            df = pd.concat([etf_data, macro_data], axis=1)
    
    # Feature Engineering: Z-Scores for macro signals
    st.write("🔧 Engineering features...")
    
    macro_cols = ['VIX', 'DXY', 'COPPER', 'GOLD', 'HY_Spread', 'T10Y2Y', 'T10Y3M', 
                  'VIX_Spot', 'VIX_3M', 'VIX_Term_Slope']
    
    for col in df.columns:
        # Create Z-scores for macro signals and volatility
        if any(m in col for m in macro_cols) or '_Vol' in col:
            # Use expanding window to avoid NaNs
            rolling_mean = df[col].rolling(60, min_periods=20).mean()
            rolling_std = df[col].rolling(60, min_periods=20).std()
            df[f"{col}_Z"] = (df[col] - rolling_mean) / (rolling_std + 1e-9)
    
    # Filter by start year and clean
    df = df[df.index.year >= start_year]
    
    # Forward fill gaps (max 5 days)
    df = df.fillna(method='ffill', limit=5)
    
    # Drop remaining NaNs
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)
    
    if dropped > 0:
        st.info(f"🧹 Dropped {dropped} rows with remaining NaNs")
    
    st.success(f"✅ Final dataset: {len(df)} rows, {len(df.columns)} columns")
    
    return df

# ------------------------------
# 6. PURE TRANSFORMER MODEL
# ------------------------------
def build_pure_transformer(input_shape, num_outputs, num_heads=4, ff_dim=128, num_layers=2, dropout_rate=0.1):
    """
    Build a pure Transformer architecture for time series prediction
    
    Args:
        input_shape: (sequence_length, num_features)
        num_outputs: Number of output predictions
        num_heads: Number of attention heads
        ff_dim: Feed-forward network dimension
        num_layers: Number of Transformer blocks
        dropout_rate: Dropout rate
    """
    inputs = Input(shape=input_shape)
    
    # Add positional encoding
    x = PositionalEncoding()(inputs)
    
    # Stack Transformer blocks
    for _ in range(num_layers):
        # Multi-head self-attention
        attn_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=input_shape[1] // num_heads,
            dropout=dropout_rate
        )(x, x)
        
        # Residual connection and layer normalization
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed-forward network
        ff_output = Dense(ff_dim, activation='relu')(x)
        ff_output = Dropout(dropout_rate)(ff_output)
        ff_output = Dense(input_shape[1])(ff_output)
        
        # Residual connection and layer normalization
        x = LayerNormalization(epsilon=1e-6)(x + ff_output)
    
    # Global pooling and output
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_outputs)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# ------------------------------
# 7. SIDEBAR CONFIGURATION
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
    
    # Manual Dataset Refresh Option
    st.subheader("📥 Dataset Management")
    force_refresh = st.checkbox(
        "Force Dataset Refresh",
        value=False,
        help="Manually fetch fresh data and update HF dataset (ignores sync window)"
    )
    
    if force_refresh:
        st.warning("⚠️ Will fetch fresh data on next run")
    
    st.divider()
    
    start_yr = st.slider("📅 Start Year", 2008, 2025, 2016)
    fee_bps = st.slider("💰 Transaction Fee (bps)", 0, 100, 15, 
                        help="Transaction cost in basis points")
    
    st.divider()
    
    st.subheader("🧠 Transformer Settings")
    epochs = st.number_input("Epochs", 10, 500, 100, step=10)
    num_heads = st.selectbox("Attention Heads", [2, 4, 8], index=1)
    num_layers = st.selectbox("Transformer Layers", [1, 2, 3, 4], index=1)
    lookback = st.slider("Lookback Days", 20, 60, 30, step=5)
    
    st.divider()
    
    st.subheader("📊 Data Split Strategy")
    split_option = st.selectbox(
        "Train/Val/Test Split",
        ["60/20/20", "70/15/15", "80/10/10"],
        index=0,
        help="Choose data split ratio: Train/Validation/Out-of-Sample"
    )
    
    # Parse split ratios
    split_ratios = {
        "60/20/20": (0.60, 0.20, 0.20),
        "70/15/15": (0.70, 0.15, 0.15),
        "80/10/10": (0.80, 0.10, 0.10)
    }
    train_pct, val_pct, test_pct = split_ratios[split_option]
    
    st.divider()
    
    run_button = st.button("🚀 Execute Transformer Alpha", type="primary", use_container_width=True)
    
    st.divider()
    
    refresh_only_button = st.button("🔄 Refresh Dataset Only", type="secondary", use_container_width=True, 
                                    help="Update HF dataset with latest data without training the model")

# ------------------------------
# 8. MAIN APPLICATION
# ------------------------------
st.title("🤖 P2-TRANSFORMER-ETF-PREDICTOR")
st.caption("Pure Transformer Architecture with Multi-Source Macro Intelligence")

# Handle dataset refresh only (no model training)
if refresh_only_button:
    st.info("🔄 Refreshing dataset without model training...")
    
    with st.status("📡 Fetching fresh data from all sources...", expanded=True):
        # Fetch fresh data
        etf_list = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
        
        st.write("📊 Fetching ETF data...")
        etf_data = fetch_etf_data(etf_list, start_date="2008-01-01")
        
        st.write("📊 Fetching macro signals...")
        macro_data = fetch_macro_data_robust(start_date="2008-01-01")
        
        if not etf_data.empty and not macro_data.empty:
            # Combine all
            new_df = pd.concat([etf_data, macro_data], axis=1)
            
            st.write("💾 Updating HuggingFace dataset...")
            token = os.getenv("HF_TOKEN")
            
            if token:
                updated_df = smart_update_hf_dataset(new_df, token)
                
                st.success("✅ Dataset refresh completed!")
                
                # Show dataset summary
                st.subheader("📊 Dataset Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Rows", len(updated_df))
                col2.metric("Total Columns", len(updated_df.columns))
                col3.metric("Date Range", f"{updated_df.index[0].strftime('%Y-%m-%d')} to {updated_df.index[-1].strftime('%Y-%m-%d')}")
                
                # Show column breakdown
                st.write("**Columns in Dataset:**")
                etf_cols = [c for c in updated_df.columns if '_Ret' in c or '_Vol' in c]
                macro_cols = [c for c in updated_df.columns if c not in etf_cols]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"📈 **ETF Signals:** {len(etf_cols)}")
                    st.code("\n".join(sorted(etf_cols)[:10]))
                
                with col2:
                    st.write(f"🌍 **Macro Signals:** {len(macro_cols)}")
                    st.code("\n".join(sorted(macro_cols)[:10]))
                
            else:
                st.error("❌ HF_TOKEN not found. Cannot update dataset.")
        else:
            st.error("❌ Failed to fetch data from sources")
    
    st.stop()  # Stop execution here, don't run model

if run_button:
    # Load and prepare data
    df = get_data(start_yr, force_refresh=force_refresh)
    
    if df.empty:
        st.error("❌ No data available. Please check data sources.")
        st.stop()
    
    # Identify target ETFs and input features
    target_etfs = [col for col in df.columns if col.endswith('_Ret') and 
                   any(etf in col for etf in ['TLT', 'TBT', 'VNQ', 'SLV', 'GLD'])]
    
    if not target_etfs:
        st.error("❌ No target ETF returns found in dataset")
        st.stop()
    
    # Input features: Z-scores and volatility
    input_features = [col for col in df.columns 
                     if (col.endswith('_Z') or col.endswith('_Vol')) 
                     and col not in target_etfs]
    
    if not input_features:
        st.error("❌ No input features found in dataset")
        st.stop()
    
    st.info(f"🎯 **Targets:** {len(target_etfs)} ETFs | **Features:** {len(input_features)} signals")
    
    # Prepare data for model
    with st.spinner("🔧 Preparing training data..."):
        scaler = MinMaxScaler()
        scaled_input = scaler.fit_transform(df[input_features])
        
        X, y = [], []
        for i in range(lookback, len(scaled_input)):
            X.append(scaled_input[i-lookback:i])
            y.append(df[target_etfs].iloc[i].values)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train/Validation/Test split based on user selection
        train_size = int(len(X) * train_pct)
        val_size = int(len(X) * val_pct)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        st.success(f"✅ Split ({split_option}): Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    
    # Build and train model
    with st.spinner("🧠 Training Pure Transformer Model..."):
        model = build_pure_transformer(
            input_shape=(X.shape[1], X.shape[2]),
            num_outputs=len(target_etfs),
            num_heads=num_heads,
            ff_dim=128,
            num_layers=num_layers,
            dropout_rate=0.1
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=int(epochs),
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        st.success(f"✅ Training completed! Best epoch: {len(history.history['loss']) - 20}")
    
    # Make predictions on test set
    with st.spinner("🔮 Generating predictions..."):
        preds = model.predict(X_test, verbose=0)
        
        # Get test dates
        test_dates = df.index[lookback + train_size + val_size:]
        
        # Get SOFR rate for Sharpe calculation
        sofr = df['T10Y3M'].iloc[-1] / 100 if 'T10Y3M' in df.columns else 0.045
        
        # Strategy execution
        strat_rets = []
        audit_trail = []
        
        for i in range(len(preds)):
            # Select best ETF based on predicted return
            best_idx = np.argmax(preds[i])
            signal_etf = target_etfs[best_idx].replace('_Ret', '')
            
            # Realized return
            realized_ret = y_test[i][best_idx]
            
            # Net return after fees
            net_ret = realized_ret - (fee_bps / 10000)
            
            strat_rets.append(net_ret)
            audit_trail.append({
                'Date': test_dates[i].strftime('%Y-%m-%d'),
                'Signal': signal_etf,
                'Predicted': preds[i][best_idx],
                'Realized': realized_ret,
                'Net_Return': net_ret
            })
        
        strat_rets = np.array(strat_rets)
    
    # ------------------------------
    # 9. PERFORMANCE ANALYTICS
    # ------------------------------
    st.divider()
    
    # Next day signal
    next_day = (df.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d')
    latest_signal = audit_trail[-1]['Signal']
    
    st.success(f"🎯 **Next Trading Day ({next_day}) Allocation:** **{latest_signal}**")
    
    st.divider()
    
    # Calculate metrics
    cum_returns = np.cumprod(1 + strat_rets)
    ann_return = (cum_returns[-1] ** (252 / len(strat_rets))) - 1
    sharpe = (np.mean(strat_rets) - (sofr / 252)) / (np.std(strat_rets) + 1e-9) * np.sqrt(252)
    
    # Hit ratio (% of positive returns in last 15 days)
    recent_rets = strat_rets[-15:]
    hit_ratio = np.mean(recent_rets > 0)
    
    # Max drawdown
    cum_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - cum_max) / cum_max
    max_dd = np.min(drawdown)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
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
        delta="Acceptable" if max_dd > -0.15 else "High Risk"
    )
    
    # Equity curve
    st.subheader("📈 Out-of-Sample Equity Curve")
    
    fig_equity = go.Figure()
    
    fig_equity.add_trace(go.Scatter(
        x=test_dates,
        y=cum_returns,
        mode='lines',
        name='Strategy',
        line=dict(color='#00d1b2', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 209, 178, 0.1)'
    ))
    
    # Add drawdown shading
    fig_equity.add_trace(go.Scatter(
        x=test_dates,
        y=cum_max,
        mode='lines',
        name='High Water Mark',
        line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dash')
    ))
    
    fig_equity.update_layout(
        template="plotly_dark",
        height=400,
        hovermode='x unified',
        showlegend=True,
        xaxis_title="Date",
        yaxis_title="Cumulative Return"
    )
    
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # Feature importance
    st.subheader("🔍 Transformer Signal Contribution (%)")
    
    # Extract attention weights from last layer
    try:
        final_weights = model.layers[-1].get_weights()[0]
        importance = np.mean(np.abs(final_weights), axis=1)
        importance_pct = (importance / importance.sum()) * 100
        
        # Map to feature names
        feat_importance = pd.Series(
            importance_pct[:len(input_features)],
            index=input_features
        ).sort_values(ascending=False).head(15)
        
        fig_imp = go.Figure(go.Bar(
            x=feat_importance.values,
            y=feat_importance.index,
            orientation='h',
            marker_color='#3b82f6',
            text=[f'{v:.1f}%' for v in feat_importance.values],
            textposition='outside'
        ))
        
        fig_imp.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Contribution %",
            yaxis_title="Feature",
            showlegend=False
        )
        
        st.plotly_chart(fig_imp, use_container_width=True)
        
    except Exception as e:
        st.warning(f"⚠️ Could not extract feature importance: {e}")
    
    # Training history
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
        yaxis_title="Loss (MSE)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_loss, use_container_width=True)
    
    # Audit trail
    st.subheader("📋 Last 15 Days Audit Trail")
    
    audit_df = pd.DataFrame(audit_trail).tail(15)
    
    def color_return(val):
        return 'color: #00ff00' if val > 0 else 'color: #ff4b4b'
    
    styled_audit = audit_df.style.applymap(
        color_return,
        subset=['Net_Return', 'Realized']
    ).format({
        'Predicted': '{:.4f}',
        'Realized': '{:.2%}',
        'Net_Return': '{:.2%}'
    })
    
    st.dataframe(styled_audit, use_container_width=True)
    
    # Model summary
    with st.expander("🔧 Model Architecture Details"):
        st.text("Pure Transformer Configuration:")
        st.text(f"  - Input Shape: ({lookback}, {len(input_features)})")
        st.text(f"  - Attention Heads: {num_heads}")
        st.text(f"  - Transformer Layers: {num_layers}")
        st.text(f"  - Feed-Forward Dim: 128")
        st.text(f"  - Output Dimension: {len(target_etfs)}")
        st.text(f"  - Total Parameters: {model.count_params():,}")
        st.text(f"  - Training Samples: {len(X_train):,}")
        st.text(f"  - Validation Samples: {len(X_val):,}")
        st.text(f"  - Test Samples: {len(X_test):,}")

else:
    st.info("👈 Configure parameters in the sidebar and click '🚀 Execute Transformer Alpha' to begin")
    
    # Show sync window info
    current_time = get_est_time()
    st.write(f"🕒 Current EST Time: **{current_time.strftime('%H:%M:%S')}**")
    
    if is_sync_window():
        st.success("✅ **Sync Window Active** - Data will be updated from sources")
    else:
        next_sync = "07:00-08:00" if current_time.hour < 7 else "19:00-20:00"
        st.info(f"⏸️ Next sync window: **{next_sync} EST**")
