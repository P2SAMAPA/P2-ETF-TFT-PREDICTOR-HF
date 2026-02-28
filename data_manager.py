"""
Data fetching, processing, and feature engineering
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime
from huggingface_hub import HfApi
import os

try:
    import pandas_datareader.data as web
    PANDAS_DATAREADER_AVAILABLE = True
except ImportError:
    PANDAS_DATAREADER_AVAILABLE = False
    st.error("Missing pandas_datareader. Please add it to requirements.txt")

from utils import get_est_time


REPO_ID = "P2SAMAPA/my-etf-data"


def fetch_macro_data_robust(start_date="2008-01-01"):
    """Fetch macro signals from multiple sources with proper error handling"""
    all_data = []
    
    # 1. FRED Data
    if PANDAS_DATAREADER_AVAILABLE:
        try:
            fred_symbols = {
                "T10Y2Y": "T10Y2Y",
                "T10Y3M": "T10Y3M",
                "DTB3":   "DTB3",        # 3-Month T-Bill — correct risk-free rate
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
    
    except Exception as e:
        st.warning(f"⚠️ VIX Term Structure failed: {e}")
    
    # Combine
    if all_data:
        combined = pd.concat(all_data, axis=1, join='outer')
        combined = combined.loc[:, ~combined.columns.duplicated()]
        combined = combined.fillna(method='ffill', limit=5)
        return combined
    else:
        st.error("❌ Failed to fetch any macro data!")
        return pd.DataFrame()


def fetch_etf_data(etfs, start_date="2008-01-01"):
    """Fetch ETF price data and calculate returns + momentum features"""
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
        
        daily_rets = etf_data.pct_change()

        # ── Daily returns (targets will be built from these) ─────────────────
        etf_returns = daily_rets.copy()
        etf_returns.columns = [f"{col}_Ret" for col in etf_returns.columns]

        # ── 20-day realized volatility ────────────────────────────────────────
        etf_vol = daily_rets.rolling(20).std() * np.sqrt(252)
        etf_vol.columns = [f"{col}_Vol" for col in etf_vol.columns]

        # ── Momentum features: rolling returns over multiple windows ──────────
        momentum_frames = []
        for window in [5, 10, 21, 63]:                    # 1W, 2W, 1M, 3M
            mom = etf_data.pct_change(window)
            mom.columns = [f"{col}_Mom{window}d" for col in mom.columns]
            momentum_frames.append(mom)

        # ── Relative strength vs SPY ──────────────────────────────────────────
        rel_frames = []
        if 'SPY' in etf_data.columns:
            spy_ret = etf_data['SPY'].pct_change(21)
            for col in etf_data.columns:
                if col != 'SPY':
                    rel = etf_data[col].pct_change(21) - spy_ret
                    rel_frames.append(rel.rename(f"{col}_RelSPY21d"))

        # ── Cross-sectional momentum rank (1=worst, 5=best among universe) ───
        target_etfs_only = [c for c in etf_data.columns
                            if c not in ['SPY', 'AGG']]
        rank_frames = []
        for window in [21, 63]:
            mom_w = etf_data[target_etfs_only].pct_change(window)
            ranked = mom_w.rank(axis=1, pct=True)
            ranked.columns = [f"{col}_Rank{window}d" for col in ranked.columns]
            rank_frames.append(ranked)

        # ── Recent trend: 5d and 10d price change ─────────────────────────────
        trend_frames = []
        for window in [5, 10]:
            trend = etf_data.pct_change(window)
            trend.columns = [f"{col}_Trend{window}d" for col in trend.columns]
            trend_frames.append(trend)

        result = pd.concat(
            [etf_returns, etf_vol] + momentum_frames +
            (rel_frames if rel_frames else []) +
            rank_frames + trend_frames,
            axis=1
        )

        return result

    except Exception as e:
        st.error(f"❌ ETF fetch failed: {e}")
        return pd.DataFrame()


def smart_update_hf_dataset(new_data, token):
    """Smart update: Only uploads if new data exists or gaps are filled.
    
    Handles new ETFs added to ETF_LIST: detects columns present in new_data
    but missing from existing HF dataset, fetches their full history, and
    backfills before merging — so the full history is populated, not just
    recent days.
    """
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

        # ── Detect newly added ETFs ───────────────────────────────────────────
        # New ETF columns will be in new_data but have all-NaN in existing_df
        # (or be completely absent). Fetch their full history and backfill.
        new_etf_cols = []
        # Infer ETF names from new_data columns ending in _Ret
        all_etfs = [c.replace("_Ret", "") for c in new_data.columns if c.endswith("_Ret")]
        for etf in all_etfs:
            ret_col = f"{etf}_Ret"
            if ret_col not in existing_df.columns or existing_df[ret_col].isna().mean() > 0.9:
                new_etf_cols.append(etf)

        if new_etf_cols:
            st.info(f"🆕 Detected new ETFs not in HF dataset: {new_etf_cols} — fetching full history...")
            # Fetch full history for new ETFs from 2008
            full_history = fetch_etf_data(new_etf_cols, start_date="2008-01-01")
            if not full_history.empty:
                if full_history.index.tz is not None:
                    full_history.index = full_history.index.tz_localize(None)
                st.success(f"✅ Full history fetched for {new_etf_cols}: "
                           f"{len(full_history)} rows from "
                           f"{full_history.index[0].date()} to "
                           f"{full_history.index[-1].date()}")
                # Merge full history into new_data by reindexing to the union of dates
                combined_index = existing_df.index.union(full_history.index)
                existing_df   = existing_df.reindex(combined_index)
                new_data      = new_data.reindex(combined_index)
                # Write new ETF columns directly into new_data across full date range
                new_cols_only = [c for c in full_history.columns
                                 if c not in existing_df.columns
                                 or existing_df[c].isna().mean() > 0.9]
                for col in new_cols_only:
                    new_data[col] = full_history.reindex(combined_index)[col]
            else:
                st.warning(f"⚠️ Could not fetch full history for {new_etf_cols}")

        combined = new_data.combine_first(existing_df)

        # Count changes — compare against original existing_df row count
        # (existing_df may have been reindexed above if new ETFs were added)
        original_row_count = len(pd.read_csv(raw_url, nrows=1)) if False else None
        new_rows    = len(combined) - len(existing_df)
        old_nulls   = existing_df.isna().sum().sum()
        new_nulls   = combined.isna().sum().sum()
        filled_gaps = old_nulls - new_nulls

        # Force upload if new ETFs were backfilled (filled_gaps may undercount
        # because existing_df was reindexed to match the new date union)
        needs_update = new_rows > 0 or filled_gaps > 0 or len(new_etf_cols) > 0
        
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
                commit_message=f"Update: {get_est_time().strftime('%Y-%m-%d %H:%M EST')} | +{new_rows} rows, filled {filled_gaps} gaps" + (f", backfilled {new_etf_cols}" if new_etf_cols else "")
            )
            
            st.success(f"✅ Dataset updated: +{new_rows} rows, filled {filled_gaps} gaps")
            return combined
        else:
            st.info("📊 Dataset already up-to-date. No upload needed.")
            return existing_df
            
    except Exception as e:
        st.warning(f"⚠️ Dataset update failed: {e}. Using new data only.")
        return new_data


def add_regime_features(df):
    """Add regime detection features"""
    
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
    
    return df


def get_data(start_year, force_refresh=False, clean_hf_dataset=False):
    """Main data fetching and processing pipeline"""
    raw_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/etf_data.csv"
    df = pd.DataFrame()
    
    # Load from HuggingFace
    try:
        df = pd.read_csv(raw_url)
        df.columns = df.columns.str.strip()
        
        date_col = next((c for c in df.columns if c.lower() in ['date', 'unnamed: 0']), df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Clean dataset if requested
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
                    st.info("📤 Uploading cleaned dataset...")
                    df.index.name = "Date"
                    df.reset_index().to_csv("etf_data.csv", index=False)
                    
                    api = HfApi()
                    api.upload_file(
                        path_or_fileobj="etf_data.csv",
                        path_in_repo="etf_data.csv",
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        token=token,
                        commit_message=f"Cleaned dataset: Removed {len(bad_cols)} columns"
                    )
                    st.success("✅ HF dataset updated!")
        
    except Exception as e:
        st.warning(f"⚠️ Could not load from HuggingFace: {e}")
    
    # Sync fresh data if needed
    from utils import is_sync_window
    should_sync = is_sync_window() or force_refresh
    
    if should_sync:
        sync_reason = "🔄 Manual Refresh" if force_refresh else "🔄 Sync Window Active"
        
        with st.status(f"{sync_reason} - Updating Dataset...", expanded=False):
            etf_list = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "SLV", "GLD", "AGG", "SPY"]
            
            etf_data = fetch_etf_data(etf_list)
            macro_data = fetch_macro_data_robust()
            
            if not etf_data.empty and not macro_data.empty:
                new_df = pd.concat([etf_data, macro_data], axis=1)
                token = os.getenv("HF_TOKEN")
                df = smart_update_hf_dataset(new_df, token)
    
    # Fetch fresh if still empty
    if df.empty:
        st.warning("📊 Fetching fresh data...")
        etf_list = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "SLV", "GLD", "AGG", "SPY"]
        etf_data = fetch_etf_data(etf_list)
        macro_data = fetch_macro_data_robust()
        
        if not etf_data.empty and not macro_data.empty:
            df = pd.concat([etf_data, macro_data], axis=1)
    
    # Feature Engineering: Z-Scores for macro + vol columns
    macro_cols = ['VIX', 'DXY', 'COPPER', 'GOLD', 'HY_Spread', 'T10Y2Y', 'T10Y3M',
                  'VIX_Spot', 'VIX_3M', 'VIX_Term_Slope']

    for col in df.columns:
        if any(m in col for m in macro_cols) or '_Vol' in col:
            rolling_mean = df[col].rolling(20, min_periods=5).mean()
            rolling_std  = df[col].rolling(20, min_periods=5).std()
            z_col = f"{col}_Z"
            df[z_col] = (df[col] - rolling_mean) / (rolling_std + 1e-9)

    # Z-score the momentum/rank/trend features too so they're on same scale
    mom_pattern_cols = [c for c in df.columns if any(
        tag in c for tag in ['_Mom', '_RelSPY', '_Rank', '_Trend']
    )]
    for col in mom_pattern_cols:
        rolling_mean = df[col].rolling(60, min_periods=10).mean()
        rolling_std  = df[col].rolling(60, min_periods=10).std()
        df[f"{col}_Z"] = (df[col] - rolling_mean) / (rolling_std + 1e-9)
    
    # Add regime features
    st.write("🎯 **Adding Regime Detection Features...**")
    df = add_regime_features(df)
    
    # Filter by start year
    df = df[df.index.year >= start_year]
    st.info(f"📅 After year filter ({start_year}+): {len(df)} samples")
    
    # Cleaning
    nan_percentages = df.isna().sum() / len(df)
    bad_features = nan_percentages[nan_percentages > 0.5].index.tolist()
    
    if bad_features:
        st.warning(f"🗑️ Dropping {len(bad_features)} features with >50% NaNs")
        df = df.drop(columns=bad_features)
    
    df = df.fillna(method='ffill', limit=5)
    df = df.fillna(method='bfill', limit=100)
    df = df.fillna(method='ffill')
    
    df = df.dropna()
    
    if len(df) > 0:
        st.success(f"✅ Final dataset: {len(df)} samples from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    
    return df
