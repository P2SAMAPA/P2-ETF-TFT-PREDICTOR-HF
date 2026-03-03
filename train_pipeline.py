"""
train_pipeline.py — Headless daily training script for GitHub Actions.

Flow:
  1. Load latest data from HF Dataset (P2SAMAPA/my-etf-data) — READ ONLY, never writes
  2. Feature engineer + train 7 Binary TFT models (80/10/10 split, hardcoded)
  3. Save outputs to root of HF Space repo (P2SAMAPA/P2-ETF-TFT-PREDICTOR):
       - model_outputs.npz   — proba scores, daily returns, dates, target_etfs
       - signals.json        — next signal, conviction, ETF scores, sofr, data range
       - training_meta.json  — lookback used, epochs, accuracy per ETF, run timestamp

app.py reads these files and replays execute_strategy() live with user sliders.
The HF dataset (my-etf-data) is NEVER touched by this script.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from io import StringIO

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log"),
    ]
)
log = logging.getLogger(__name__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = "42"

HF_SPACE_REPO = "P2SAMAPA/P2-ETF-TFT-PREDICTOR"
HF_DATASET_REPO = "P2SAMAPA/my-etf-data"   # READ ONLY — never written by this script

# Hardcoded split
TRAIN_PCT = 0.80
VAL_PCT   = 0.10
TEST_PCT  = 0.10
FORWARD_DAYS = 5


# ── Streamlit mock — allows importing data_manager.py headlessly ──────────────
def _make_st_mock():
    import unittest.mock as mock
    st = mock.MagicMock()
    st.warning = lambda *a, **k: log.warning(" ".join(str(x) for x in a))
    st.error   = lambda *a, **k: log.error(" ".join(str(x) for x in a))
    st.info    = lambda *a, **k: log.info(" ".join(str(x) for x in a))
    st.success = lambda *a, **k: log.info(" ".join(str(x) for x in a))
    st.write   = lambda *a, **k: log.info(" ".join(str(x) for x in a))
    cm = mock.MagicMock()
    cm.__enter__ = lambda s: s
    cm.__exit__  = mock.MagicMock(return_value=False)
    st.status  = mock.MagicMock(return_value=cm)
    st.spinner = mock.MagicMock(return_value=cm)
    st.secrets = {}
    return st

sys.modules["streamlit"] = _make_st_mock()


def push_file_to_hf_space(filename: str, content_bytes: bytes,
                           commit_msg: str, token: str):
    """Push a file to the HF Space repo root."""
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    ops = [CommitOperationAdd(path_in_repo=filename,
                               path_or_fileobj=content_bytes)]
    print(f"Pushing {filename} to repo_id={HF_SPACE_REPO} repo_type=space")                        
    api.create_commit(
        repo_id=HF_SPACE_REPO,
        repo_type="space",
        token=token,
        commit_message=commit_msg,
        operations=ops,
    )
    log.info(f"✅ Pushed {filename} → {HF_SPACE_REPO} ({len(content_bytes):,} bytes)")


def fetch_sofr():
    """Fetch latest 3M T-Bill rate from FRED."""
    try:
        import pandas_datareader.data as web
        dtb3 = web.DataReader('DTB3', 'fred', start='2024-01-01').dropna()
        if not dtb3.empty:
            rate = float(dtb3.iloc[-1].values[0]) / 100
            date = dtb3.index[-1].date()
            log.info(f"Risk-free rate: {rate*100:.2f}% (FRED DTB3 {date})")
            return rate, f"FRED DTB3 ({date})"
    except Exception as e:
        log.warning(f"FRED fetch failed: {e}")
    return 0.045, "fallback 4.5%"


def main(force_refresh: bool = False):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable not set")

    # ── 1. Import project modules (streamlit already mocked) ─────────────────
    from data_manager import get_data
    from models import (
        build_binary_tft, train_all_binary_tfts,
        predict_binary_tfts, SEED
    )
    from strategy import execute_strategy
    from sklearn.preprocessing import RobustScaler
    import tensorflow as tf
    import random

    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

    # ── 2. Load dataset (reads P2SAMAPA/my-etf-data, never writes it) ────────
    log.info("Loading dataset from HF...")
    df = get_data(start_year=2008, force_refresh=force_refresh,
                  clean_hf_dataset=False)
    if df is None or df.empty:
        raise RuntimeError("Dataset is empty — aborting")
    log.info(f"Dataset: {len(df)} rows × {df.shape[1]} cols | "
             f"{df.index[0].date()} → {df.index[-1].date()}")

    # ── 3. Identify targets and features ─────────────────────────────────────
    TARGET_ETF_LABELS = ['TLT', 'VCIT', 'LQD', 'HYG', 'VNQ', 'SLV', 'GLD']
    target_etfs = [c for c in df.columns
                   if c.endswith('_Ret') and any(e in c for e in TARGET_ETF_LABELS)]
    input_features = [c for c in df.columns
                      if (c.endswith('_Z') or c.endswith('_Vol') or
                          'Regime' in c or 'YC_' in c or 'Credit_' in c or
                          'Rates_' in c or 'VIX_Term_' in c or
                          'Rising' in c or 'Falling' in c or 'Accelerating' in c)
                      and c not in target_etfs]

    if not target_etfs or not input_features:
        raise RuntimeError(f"Missing targets ({len(target_etfs)}) or "
                           f"features ({len(input_features)})")
    log.info(f"Targets: {len(target_etfs)} | Features: {len(input_features)}")

    # ── 4. Risk-free rate ─────────────────────────────────────────────────────
    sofr, rf_label = fetch_sofr()
    if 'DTB3' in df.columns and 'fallback' in rf_label:
        sofr = float(df['DTB3'].dropna().iloc[-1]) / 100
        rf_label = "dataset DTB3"

    # ── 5. Forward return targets ─────────────────────────────────────────────
    daily_rf_5d = (sofr / 252) * FORWARD_DAYS
    fwd_returns = pd.DataFrame(index=df.index)
    for col in target_etfs:
        fwd_returns[col] = df[col].rolling(FORWARD_DAYS).sum().shift(-FORWARD_DAYS)

    valid_idx = fwd_returns.dropna().index
    df_model  = df.loc[valid_idx]
    fwd_model = fwd_returns.loc[valid_idx]

    binary_targets = pd.DataFrame(index=df_model.index)
    for col in target_etfs:
        binary_targets[col] = (fwd_model[col] > daily_rf_5d).astype(np.int32)

    # ── 6. Scale features (fit on train only) ────────────────────────────────
    approx_train_end = int((len(df_model) - 60 - 1) * TRAIN_PCT)
    scaler = RobustScaler()
    scaler.fit(df_model[input_features].values[:approx_train_end])
    scaled = scaler.transform(df_model[input_features].values)

    # ── 7. Auto-optimise lookback ─────────────────────────────────────────────
    LOOKBACK_CANDIDATES = [20, 30, 40, 50, 60]
    best_lookback, best_val_loss = 30, float('inf')
    lookback_results = {}
    bin_proxy = binary_targets[target_etfs[0]].values

    log.info("Auto-optimising lookback window...")
    for lb in LOOKBACK_CANDIDATES:
        X_lb, y_lb = [], []
        for i in range(lb, len(scaled) - 1):
            X_lb.append(scaled[i - lb:i])
            y_lb.append(bin_proxy[i + 1])
        X_lb = np.array(X_lb, dtype=np.float32)
        y_lb = np.array(y_lb, dtype=np.float32)

        ts = int(len(X_lb) * TRAIN_PCT)
        vs = int(len(X_lb) * VAL_PCT)

        random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
        probe = build_binary_tft(seq_len=lb, num_features=X_lb.shape[2])
        probe.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
                      loss='binary_crossentropy')
        h = probe.fit(
            X_lb[:ts], y_lb[:ts],
            validation_data=(X_lb[ts:ts+vs], y_lb[ts:ts+vs]),
            epochs=20, batch_size=64,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True)],
            verbose=0
        )
        vl = min(h.history['val_loss'])
        lookback_results[lb] = round(vl, 4)
        log.info(f"  lookback={lb}d → val_loss={vl:.4f}")
        if vl < best_val_loss:
            best_val_loss = vl
            best_lookback = lb

    lookback = best_lookback
    log.info(f"Best lookback: {lookback}d")

    # ── 8. Build sequences with optimal lookback + T+1 shift ─────────────────
    bin_vals       = binary_targets[target_etfs].values
    fwd_vals       = fwd_model[target_etfs].values
    df_model_dates = df_model.index

    X, y_bin, y_fwd, dates_seq = [], [], [], []
    for i in range(lookback, len(scaled) - 1):
        X.append(scaled[i - lookback:i])
        y_bin.append(bin_vals[i + 1])
        y_fwd.append(fwd_vals[i + 1])
        dates_seq.append(df_model_dates[i + 1])

    X      = np.array(X,     dtype=np.float32)
    y_bin  = np.array(y_bin, dtype=np.int32)
    y_fwd  = np.array(y_fwd, dtype=np.float32)
    dates_seq = pd.DatetimeIndex(dates_seq)

    train_size = int(len(X) * TRAIN_PCT)
    val_size   = int(len(X) * VAL_PCT)

    X_train     = X[:train_size]
    y_bin_train = y_bin[:train_size]
    X_val       = X[train_size:train_size + val_size]
    y_bin_val   = y_bin[train_size:train_size + val_size]
    X_test      = X[train_size + val_size:]
    y_fwd_test  = y_fwd[train_size + val_size:]
    y_bin_test  = y_bin[train_size + val_size:]
    test_dates  = dates_seq[train_size + val_size:]

    log.info(f"Split → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── 9. Train 7 binary TFTs ────────────────────────────────────────────────
    etf_names = [e.replace('_Ret', '') for e in target_etfs]
    log.info(f"Training {len(target_etfs)} Binary TFTs...")
    models, histories = train_all_binary_tfts(
        X_train, y_bin_train, X_val, y_bin_val,
        etf_names=etf_names, epochs=150
    )
    epochs_per = {n: len(h.history['loss'])
                  for n, h in zip(etf_names, histories)}
    log.info(f"Training complete — epochs: {epochs_per}")

    # ── 10. Predict ───────────────────────────────────────────────────────────
    proba = predict_binary_tfts(models, X_test)   # (N_test, n_etfs)

    # Accuracy per ETF
    acc_per_etf = {}
    for j, name in enumerate(etf_names):
        preds_j = (proba[:, j] > 0.5).astype(int)
        acc_per_etf[name] = round(float(np.mean(preds_j == y_bin_test[:, j])), 4)
    log.info(f"Binary accuracy: {acc_per_etf}")

    # ── 11. Daily returns on test set (for live strategy replay in app) ───────
    daily_ret_test = df.reindex(test_dates)[target_etfs].fillna(0.0).values
    # Also save full benchmark returns on test period
    spy_ret_test = df.reindex(test_dates)['SPY_Ret'].fillna(0.0).values \
                   if 'SPY_Ret' in df.columns else np.zeros(len(test_dates))
    agg_ret_test = df.reindex(test_dates)['AGG_Ret'].fillna(0.0).values \
                   if 'AGG_Ret' in df.columns else np.zeros(len(test_dates))

    # ── 12. Compute next-day signal (from last row of proba) ──────────────────
    from strategy import compute_signal_conviction
    from utils import get_next_trading_day
    last_scores = proba[-1]
    best_idx, conviction_z, conviction_label = compute_signal_conviction(last_scores)
    next_signal = etf_names[best_idx]
    next_date   = get_next_trading_day(test_dates[-1])

    # ── 13. Save outputs ──────────────────────────────────────────────────────
    log.info("Saving outputs...")

    # --- model_outputs.npz ---
    # Contains everything app.py needs to replay execute_strategy() live
    npz_buf = {}
    npz_buf['proba']          = proba.astype(np.float32)       # (N, 7)
    npz_buf['daily_ret_test'] = daily_ret_test.astype(np.float32)  # (N, 7)
    npz_buf['y_fwd_test']     = y_fwd_test.astype(np.float32)  # (N, 7)
    npz_buf['spy_ret_test']   = spy_ret_test.astype(np.float32)  # (N,)
    npz_buf['agg_ret_test']   = agg_ret_test.astype(np.float32)  # (N,)
    npz_buf['test_dates']     = np.array([str(d.date()) for d in test_dates])
    npz_buf['target_etfs']    = np.array(target_etfs)
    npz_buf['sofr']           = np.array([sofr])
    # Full date range info for start_year filter
    npz_buf['all_test_start'] = np.array([str(test_dates[0].date())])
    npz_buf['all_test_end']   = np.array([str(test_dates[-1].date())])

    import io as _io
    npz_io = _io.BytesIO()
    np.savez_compressed(npz_io, **npz_buf)
    npz_bytes = npz_io.getvalue()

    # --- signals.json ---
    signals_payload = {
        "next_signal":       next_signal,
        "next_date":         str(next_date),
        "conviction_z":      round(float(conviction_z), 4),
        "conviction_label":  conviction_label,
        "etf_scores":        {name: round(float(score), 4)
                              for name, score in zip(etf_names, last_scores)},
        "sofr":              round(sofr, 6),
        "rf_label":          rf_label,
        "data_start":        str(df.index[0].date()),
        "data_end":          str(df.index[-1].date()),
        "test_start":        str(test_dates[0].date()),
        "test_end":          str(test_dates[-1].date()),
        "n_test_days":       int(len(test_dates)),
        "lookback_days":     int(lookback),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # --- training_meta.json ---
    meta_payload = {
        "lookback_days":     int(lookback),
        "lookback_search":   lookback_results,
        "epochs_per_etf":    epochs_per,
        "accuracy_per_etf":  acc_per_etf,
        "train_size":        int(train_size),
        "val_size":          int(val_size),
        "test_size":         int(len(X_test)),
        "n_features":        int(len(input_features)),
        "n_targets":         int(len(target_etfs)),
        "target_etfs":       target_etfs,
        "split":             "80/10/10",
        "forward_days":      FORWARD_DAYS,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # ── 14. Push all outputs to HF Space repo ────────────────────────────────
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    push_file_to_hf_space(
        "model_outputs.npz",
        npz_bytes,
        f"Update model outputs {run_date}",
        token
    )
    push_file_to_hf_space(
        "signals.json",
        json.dumps(signals_payload, indent=2).encode(),
        f"Update signals {run_date} → {next_signal}",
        token
    )
    push_file_to_hf_space(
        "training_meta.json",
        json.dumps(meta_payload, indent=2).encode(),
        f"Update training meta {run_date}",
        token
    )

    log.info(f"✅ All outputs pushed to {HF_SPACE_REPO}")
    log.info(f"📡 Next signal: {next_signal} on {next_date} "
             f"(conviction: {conviction_label}, Z={conviction_z:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force full dataset rebuild")
    args = parser.parse_args()
    main(force_refresh=args.force_refresh)
