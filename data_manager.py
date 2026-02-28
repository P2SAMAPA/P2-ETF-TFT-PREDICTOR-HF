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

# ── ETF universe ──────────────────────────────────────────────────────────────
# TBT removed (leveraged decay). Added: VCIT, LQD, HYG (investment grade + HY credit)
ETF_LIST     = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "SLV", "GLD", "AGG", "SPY"]
TARGET_ETFS  = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "SLV", "GLD"]   # excludes benchmarks


def fetch_macro_data_robust(start_date="2008-01-01"):
    """Fetch macro signals from FRED and Yahoo Finance"""
    all_data = []

    # 1. FRED
    if PANDAS_DATAREADER_AVAILABLE:
        try:
            fred_symbols = {
                "T10Y2Y":       "T10Y2Y",
                "T10Y3M":       "T10Y3M",
                "DTB3":         "DTB3",
                "BAMLH0A0HYM2": "HY_Spread",
                "VIXCLS":       "VIX",
                "DTWEXBGS":     "DXY",
            }
            fred_data = web.DataReader(
                list(fred_symbols.keys()), "fred", start_date, datetime.now()
            )
            fred_data.columns = [fred_symbols[col] for col in fred_data.columns]
            if fred_data.index.tz is not None:
                fred_data.index = fred_data.index.tz_localize(None)
            all_data.append(fred_data)
        except Exception as e:
            st.warning(f"⚠️ FRED partial failure: {e}")

    # 2. Yahoo Finance — gold, copper, VIX
    try:
        yf_symbols = {"GC=F": "GOLD", "HG=F": "COPPER", "^VIX": "VIX_YF"}
        yf_data = yf.download(
            list(yf_symbols.keys()), start=start_date, progress=False, auto_adjust=True
        )["Close"]
        if isinstance(yf_data, pd.Series):
            yf_data = yf_data.to_frame()
        yf_data.columns = [yf_symbols.get(col, col) for col in yf_data.columns]
        if yf_data.index.tz is not None:
            yf_data.index = yf_data.index.tz_localize(None)
        all_data.append(yf_data)
    except Exception as e:
        st.warning(f"⚠️ Yahoo Finance failed: {e}")

    # 3. VIX term structure
    try:
        vix_term = yf.download(
            ["^VIX", "^VIX3M"], start=start_date, progress=False, auto_adjust=True
        )["Close"]
        if not vix_term.empty:
            if isinstance(vix_term, pd.Series):
                vix_term = vix_term.to_frame()
            vix_term.columns = ["VIX_Spot", "VIX_3M"]
            vix_term["VIX_Term_Slope"] = vix_term["VIX_3M"] - vix_term["VIX_Spot"]
            if vix_term.index.tz is not None:
                vix_term.index = vix_term.index.tz_localize(None)
            all_data.append(vix_term)
    except Exception as e:
        st.warning(f"⚠️ VIX Term Structure failed: {e}")

    if all_data:
        combined = pd.concat(all_data, axis=1, join="outer")
        combined = combined.loc[:, ~combined.columns.duplicated()]
        combined = combined.ffill(limit=5)
        return combined
    else:
        st.error("❌ Failed to fetch any macro data!")
        return pd.DataFrame()


def fetch_etf_data(etfs, start_date="2008-01-01"):
    """
    Fetch ETF price data and calculate features.

    Produces exactly 3 columns per ETF to match HF dataset schema:
        {ETF}_Ret   — daily return
        {ETF}_MA20  — 20-day simple moving average of price (raw)
        {ETF}_Vol   — 20-day annualised realised volatility
    """
    try:
        etf_data = yf.download(
            etfs, start=start_date, progress=False, auto_adjust=True
        )["Close"]

        if isinstance(etf_data, pd.Series):
            etf_data = etf_data.to_frame()

        if etf_data.index.tz is not None:
            etf_data.index = etf_data.index.tz_localize(None)

        daily_rets = etf_data.pct_change()

        # Daily returns
        etf_returns = daily_rets.copy()
        etf_returns.columns = [f"{col}_Ret" for col in etf_returns.columns]

        # 20-day simple moving average (raw price)
        etf_ma20 = etf_data.rolling(20).mean()
        etf_ma20.columns = [f"{col}_MA20" for col in etf_ma20.columns]

        # 20-day annualised realised volatility
        etf_vol = daily_rets.rolling(20).std() * np.sqrt(252)
        etf_vol.columns = [f"{col}_Vol" for col in etf_vol.columns]

        result = pd.concat([etf_returns, etf_ma20, etf_vol], axis=1)
        return result

    except Exception as e:
        st.error(f"❌ ETF fetch failed: {e}")
        return pd.DataFrame()


def smart_update_hf_dataset(new_data, token, force_upload=False):
    """
    Smart update: merges new_data on top of existing HF dataset and uploads
    if anything changed — or always uploads when force_upload=True.

    Also handles newly added ETFs: detects ETFs whose _Ret column is missing
    or all-NaN in the existing dataset, fetches their full history back to
    2008, and backfills before uploading.
    """
    if not token:
        st.warning("⚠️ No HF_TOKEN found. Skipping dataset update.")
        return new_data

    raw_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/etf_data.csv"

    try:
        # Cache-bust the HF CDN so we always read the latest version
        import time
        bust_url = f"{raw_url}?t={int(time.time())}"
        existing_df = pd.read_csv(bust_url)
        existing_df.columns = existing_df.columns.str.strip()
        date_col = next(
            (c for c in existing_df.columns if c.lower() in ["date", "unnamed: 0"]),
            existing_df.columns[0],
        )
        existing_df[date_col] = pd.to_datetime(existing_df[date_col])
        existing_df = existing_df.set_index(date_col).sort_index()
        if existing_df.index.tz is not None:
            existing_df.index = existing_df.index.tz_localize(None)

        # ── Step 1: fetch FULL ETF history from 2008 ───────────────────────
        st.info("📡 Fetching full ETF history from 2008...")
        full_etf = fetch_etf_data(ETF_LIST, start_date="2008-01-01")
        if full_etf.index.tz is not None:
            full_etf.index = full_etf.index.tz_localize(None)
        st.info(f"📊 Full ETF fetch: {len(full_etf)} rows, columns: {list(full_etf.columns)}")

        # Detect new ETFs for reporting
        new_etf_cols = [
            etf for etf in ETF_LIST
            if f"{etf}_Ret" not in existing_df.columns
            or existing_df[f"{etf}_Ret"].isna().mean() > 0.9
        ]
        if new_etf_cols:
            st.info(f"🆕 New ETFs detected: {new_etf_cols}")

        # ── Step 2: extract macro columns from existing_df ───────────────────
        # Macro cols = everything in existing_df that is NOT an ETF column
        etf_col_names = [c for c in existing_df.columns
                         if any(c.startswith(f"{e}_") for e in
                                ["TLT","TBT","VCIT","LQD","HYG","VNQ","SLV","GLD","AGG","SPY"])]
        macro_col_names = [c for c in existing_df.columns if c not in etf_col_names]
        macro_existing  = existing_df[macro_col_names].copy()
        st.info(f"📊 Macro cols from existing: {macro_col_names}")

        # ── Step 3: build combined from scratch using pd.concat ──────────────
        # ETF data: full_etf (authoritative, full history)
        # Macro data: new_data macro cols combined_first with existing macro
        macro_new_cols = [c for c in new_data.columns
                          if not any(c.startswith(f"{e}_") for e in
                                     ["TLT","TBT","VCIT","LQD","HYG","VNQ","SLV","GLD","AGG","SPY"])]
        macro_new = new_data[macro_new_cols].copy() if macro_new_cols else pd.DataFrame()

        if not macro_new.empty:
            macro_combined = macro_new.combine_first(macro_existing)
        else:
            macro_combined = macro_existing

        # Align on union of all dates
        full_index = full_etf.index.union(macro_combined.index)
        etf_aligned   = full_etf.reindex(full_index)
        macro_aligned = macro_combined.reindex(full_index)

        combined = pd.concat([etf_aligned, macro_aligned], axis=1)
        st.info(f"📊 Combined shape: {combined.shape} | ETF cols: {len(full_etf.columns)} | Macro cols: {len(macro_col_names)}")

        # ── Step 4: decide whether to upload ─────────────────────────────────
        new_rows    = len(combined) - len(existing_df)
        old_nulls   = existing_df.isna().sum().sum()
        new_nulls   = combined.isna().sum().sum()
        filled_gaps = old_nulls - new_nulls
        needs_update = force_upload or new_rows > 0 or filled_gaps > 0 or len(new_etf_cols) > 0

        if needs_update:
            combined.index.name = "Date"
            # Verify new ETFs are in combined before writing
            missing = [e for e in ["VCIT","LQD","HYG"] if f"{e}_Ret" not in combined.columns]
            present = [e for e in ["VCIT","LQD","HYG"] if f"{e}_Ret" in combined.columns]
            st.info(f"📋 Pre-upload check — new ETFs present: {present} | missing: {missing}")
            if present:
                sample = combined[[f"{e}_Ret" for e in present]].dropna().head(3)
                st.info(f"📋 Sample data:\n{sample.to_string()}")
            out_df = combined.reset_index()
            st.info(f"📋 CSV will have {len(out_df)} rows, {len(out_df.columns)} columns")
            # Write to a unique filename to avoid any file caching issues
            import time
            csv_filename = f"etf_data_{int(time.time())}.csv"
            out_df.to_csv(csv_filename, index=False)
            # Verify file on disk before upload
            verify = pd.read_csv(csv_filename, nrows=5)
            new_etf_check = [c for c in ["VCIT_Ret","LQD_Ret","HYG_Ret"] if c in verify.columns]
            st.info(f"📋 File on disk check — new ETF cols: {new_etf_check}, shape: {verify.shape}")
            api = HfApi()
            api.upload_file(
                path_or_fileobj=csv_filename,
                path_in_repo="etf_data.csv",
                repo_id=REPO_ID,
                repo_type="dataset",
                token=token,
                commit_message=(
                    ("FORCE " if force_upload else "") +
                    f"Update: {get_est_time().strftime('%Y-%m-%d %H:%M EST')} | "
                    f"+{new_rows} rows, filled {filled_gaps} gaps" +
                    (f", backfilled {new_etf_cols}" if new_etf_cols else "")
                ),
            )
            st.success(f"✅ Dataset updated: +{new_rows} rows, filled {filled_gaps} gaps"
                       + (f", backfilled {new_etf_cols}" if new_etf_cols else ""))
            return combined
        else:
            st.info("📊 Dataset already up-to-date. No upload needed.")
            return existing_df

    except Exception as e:
        st.warning(f"⚠️ Dataset update failed: {e}. Using new data only.")
        return new_data


def add_regime_features(df):
    """Add regime detection features using pd.concat to avoid fragmentation"""
    new_cols = {}

    if "VIX" in df.columns:
        new_cols["VIX_Regime_Low"]  = (df["VIX"] < 15).astype(int)
        new_cols["VIX_Regime_Med"]  = ((df["VIX"] >= 15) & (df["VIX"] < 25)).astype(int)
        new_cols["VIX_Regime_High"] = (df["VIX"] >= 25).astype(int)

    if "T10Y2Y" in df.columns:
        new_cols["YC_Inverted"] = (df["T10Y2Y"] < 0).astype(int)
        new_cols["YC_Flat"]     = ((df["T10Y2Y"] >= 0) & (df["T10Y2Y"] < 0.5)).astype(int)
        new_cols["YC_Steep"]    = (df["T10Y2Y"] >= 0.5).astype(int)

    if "HY_Spread" in df.columns:
        new_cols["Credit_Stress_Low"]  = (df["HY_Spread"] < 400).astype(int)
        new_cols["Credit_Stress_Med"]  = ((df["HY_Spread"] >= 400) & (df["HY_Spread"] < 600)).astype(int)
        new_cols["Credit_Stress_High"] = (df["HY_Spread"] >= 600).astype(int)

    if "VIX_Term_Slope" in df.columns:
        new_cols["VIX_Term_Contango"]      = (df["VIX_Term_Slope"] > 2).astype(int)
        new_cols["VIX_Term_Backwardation"] = (df["VIX_Term_Slope"] < -2).astype(int)

    if "T10Y3M" in df.columns:
        new_cols["Rates_VeryLow"] = (df["T10Y3M"] < 1.0).astype(int)
        new_cols["Rates_Low"]     = ((df["T10Y3M"] >= 1.0) & (df["T10Y3M"] < 2.0)).astype(int)
        new_cols["Rates_Normal"]  = ((df["T10Y3M"] >= 2.0) & (df["T10Y3M"] < 3.0)).astype(int)
        new_cols["Rates_High"]    = (df["T10Y3M"] >= 3.0).astype(int)

    if "T10Y2Y" in df.columns:
        yc_mom20 = df["T10Y2Y"].diff(20)
        yc_mom60 = df["T10Y2Y"].diff(60)
        new_cols["YC_Mom20d"]        = yc_mom20
        new_cols["YC_Mom60d"]        = yc_mom60
        new_cols["Rates_Rising20d"]  = (yc_mom20 > 0).astype(int)
        new_cols["Rates_Falling20d"] = (yc_mom20 < 0).astype(int)
        new_cols["Rates_Rising60d"]  = (yc_mom60 > 0).astype(int)
        new_cols["Rates_Falling60d"] = (yc_mom60 < 0).astype(int)
        yc_accel = yc_mom20.diff(20)
        new_cols["YC_Accel"]           = yc_accel
        new_cols["Rates_Accelerating"] = (yc_accel > 0).astype(int)

    if "T10Y3M" in df.columns:
        t3m_mom20 = df["T10Y3M"].diff(20)
        t3m_mom60 = df["T10Y3M"].diff(60)
        new_cols["T10Y3M_Mom20d"]     = t3m_mom20
        new_cols["T10Y3M_Mom60d"]     = t3m_mom60
        new_cols["T10Y3M_Rising20d"]  = (t3m_mom20 > 0).astype(int)
        new_cols["T10Y3M_Falling20d"] = (t3m_mom20 < 0).astype(int)
        new_cols["T10Y3M_Rising60d"]  = (t3m_mom60 > 0).astype(int)
        new_cols["T10Y3M_Falling60d"] = (t3m_mom60 < 0).astype(int)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


def get_data(start_year, force_refresh=False, clean_hf_dataset=False):
    """Main data fetching and processing pipeline"""
    raw_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/etf_data.csv"
    df = pd.DataFrame()

    # ── Load from HuggingFace ─────────────────────────────────────────────────
    try:
        import time
        df = pd.read_csv(f"{raw_url}?t={int(time.time())}")
        df.columns = df.columns.str.strip()
        date_col = next(
            (c for c in df.columns if c.lower() in ["date", "unnamed: 0"]), df.columns[0]
        )
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Optional: clean >30% NaN columns
        if clean_hf_dataset:
            st.warning("🧹 **Cleaning HF Dataset Mode Active**")
            original_cols = len(df.columns)
            nan_pct  = (df.isna().sum() / len(df)) * 100
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
                        path_or_fileobj=csv_filename,
                        path_in_repo="etf_data.csv",
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        token=token,
                        commit_message=f"Cleaned dataset: Removed {len(bad_cols)} columns",
                    )
                    st.success("✅ HF dataset updated!")

    except Exception as e:
        st.warning(f"⚠️ Could not load from HuggingFace: {e}")

    # ── Sync / force refresh ──────────────────────────────────────────────────
    from utils import is_sync_window
    should_sync = is_sync_window() or force_refresh

    if should_sync:
        sync_reason = "🔄 Manual Refresh" if force_refresh else "🔄 Sync Window Active"
        with st.status(f"{sync_reason} - Updating Dataset...", expanded=False):
            etf_data   = fetch_etf_data(ETF_LIST)
            macro_data = fetch_macro_data_robust()
            if not etf_data.empty and not macro_data.empty:
                new_df = pd.concat([etf_data, macro_data], axis=1)
                token  = os.getenv("HF_TOKEN")
                df     = smart_update_hf_dataset(new_df, token, force_upload=force_refresh)

    # ── Fallback: fetch fresh if still empty ──────────────────────────────────
    if df.empty:
        st.warning("📊 Fetching fresh data...")
        etf_data   = fetch_etf_data(ETF_LIST)
        macro_data = fetch_macro_data_robust()
        if not etf_data.empty and not macro_data.empty:
            df = pd.concat([etf_data, macro_data], axis=1)

    # ── Feature engineering: Z-scores ────────────────────────────────────────
    macro_cols = [
        "VIX", "DXY", "COPPER", "GOLD", "HY_Spread", "T10Y2Y", "T10Y3M",
        "VIX_Spot", "VIX_3M", "VIX_Term_Slope",
    ]
    for col in df.columns:
        if any(m in col for m in macro_cols) or "_Vol" in col:
            roll_mean = df[col].rolling(20, min_periods=5).mean()
            roll_std  = df[col].rolling(20, min_periods=5).std()
            df[f"{col}_Z"] = (df[col] - roll_mean) / (roll_std + 1e-9)

    # ── Regime features ───────────────────────────────────────────────────────
    st.write("🎯 **Adding Regime Detection Features...**")
    df = add_regime_features(df)

    # ── Filter by start year ──────────────────────────────────────────────────
    df = df[df.index.year >= start_year]
    st.info(f"📅 After year filter ({start_year}+): {len(df)} samples")

    # ── Drop columns with >50% NaNs ───────────────────────────────────────────
    nan_pct      = df.isna().sum() / len(df)
    bad_features = nan_pct[nan_pct > 0.5].index.tolist()
    if bad_features:
        st.warning(f"🗑️ Dropping {len(bad_features)} features with >50% NaNs")
        df = df.drop(columns=bad_features)

    # ── Fill remaining NaNs ───────────────────────────────────────────────────
    df = df.ffill(limit=5).bfill(limit=100).ffill()
    df = df.dropna()

    if len(df) > 0:
        st.success(
            f"✅ Final dataset: {len(df)} samples from "
            f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
        )

    return df
