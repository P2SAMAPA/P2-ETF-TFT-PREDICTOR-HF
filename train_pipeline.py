"""
train_pipeline.py — Headless training script for GitHub Actions.

Supports:
- Per-year model training (existing, default)
- Global model training (--mode train-global)
- Global model prediction for a specific year (--mode predict-global --year YYYY)

Outputs:
- Per-year model: saved to option_{option}/model_outputs.npz, signals.json, training_meta.json
  and sweep files in sweep/option_{option}/signals_{year}_{date}.json
- Global model: saved to option_{option}/global_model/ (model.h5, scaler.pkl, meta.json)
- Global predictions: saved to global_sweep/option_{option}/signals_{year}_{date}.json
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import io
import pickle
import random
import tempfile

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

HF_OUTPUT_REPO  = "P2SAMAPA/p2-etf-tft-outputs"   # dataset repo — all outputs go here
HF_DATASET_REPO = "P2SAMAPA/my-etf-data"           # READ ONLY — never written

TRAIN_PCT    = 0.80
VAL_PCT      = 0.10
TEST_PCT     = 0.10
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

# ── HF utilities ──────────────────────────────────────────────────────────────
def push_file_to_hf_dataset(filename: str, content_bytes: bytes,
                            commit_msg: str, token: str):
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    ops = [CommitOperationAdd(path_in_repo=filename,
                               path_or_fileobj=content_bytes)]
    api.create_commit(
        repo_id=HF_OUTPUT_REPO,
        repo_type="dataset",
        token=token,
        commit_message=commit_msg,
        operations=ops,
    )
    log.info(f"✅ Pushed {filename} → {HF_OUTPUT_REPO} ({len(content_bytes):,} bytes)")

def download_file_from_hf_dataset(filename: str, token: str):
    from huggingface_hub import hf_hub_download
    local_path = hf_hub_download(
        repo_id=HF_OUTPUT_REPO,
        repo_type="dataset",
        filename=filename,
        token=token,
        local_dir=None,
    )
    log.info(f"Downloaded {filename} to {local_path}")
    return local_path

# ── Data fetching and preparation (reused) ───────────────────────────────────
def fetch_sofr():
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

def prepare_data(option: str, start_year: int, force_refresh: bool):
    from config import OPTION_A_ETFS, OPTION_B_ETFS
    from data_manager import get_data

    if option == 'a':
        TARGET_ETF_LABELS = OPTION_A_ETFS
    else:
        TARGET_ETF_LABELS = OPTION_B_ETFS

    log.info(f"Loading dataset from HF, start_year={start_year}...")
    df = get_data(start_year=start_year, force_refresh=force_refresh,
                  clean_hf_dataset=False)
    if df is None or df.empty:
        raise RuntimeError("Dataset is empty — aborting")
    log.info(f"Dataset: {len(df)} rows × {df.shape[1]} cols | "
             f"{df.index[0].date()} → {df.index[-1].date()}")

    # Identify targets and features
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

    # Risk-free rate
    sofr, rf_label = fetch_sofr()
    if 'DTB3' in df.columns and 'fallback' in rf_label:
        sofr = float(df['DTB3'].dropna().iloc[-1]) / 100
        rf_label = "dataset DTB3"

    # Forward returns
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

    return df, df_model, fwd_model, binary_targets, target_etfs, input_features, sofr, rf_label

def optimize_lookback(scaled, bin_proxy, lookback_candidates, seed):
    import tensorflow as tf
    from models import build_binary_tft
    best_lookback, best_val_loss = 30, float('inf')
    lookback_results = {}
    for lb in lookback_candidates:
        X_lb, y_lb = [], []
        for i in range(lb, len(scaled) - 1):
            X_lb.append(scaled[i - lb:i])
            y_lb.append(bin_proxy[i + 1])
        X_lb = np.array(X_lb, dtype=np.float32)
        y_lb = np.array(y_lb, dtype=np.float32)

        ts = int(len(X_lb) * TRAIN_PCT)
        vs = int(len(X_lb) * VAL_PCT)

        random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
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
    log.info(f"Best lookback: {best_lookback}d")
    return best_lookback, lookback_results

def create_sequences(scaled, bin_vals, fwd_vals, dates, lookback, seed):
    import tensorflow as tf
    np.random.seed(seed); tf.random.set_seed(seed)

    X, y_bin, y_fwd, dates_seq = [], [], [], []
    for i in range(lookback, len(scaled) - 1):
        X.append(scaled[i - lookback:i])
        y_bin.append(bin_vals[i + 1])
        y_fwd.append(fwd_vals[i + 1])
        dates_seq.append(dates[i + 1])

    X      = np.array(X, dtype=np.float32)
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

    return X_train, y_bin_train, X_val, y_bin_val, X_test, y_fwd_test, y_bin_test, test_dates

def train_models(X_train, y_bin_train, X_val, y_bin_val, etf_names, epochs=150, seed=42):
    from models import train_all_binary_tfts
    import tensorflow as tf
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    models, histories = train_all_binary_tfts(
        X_train, y_bin_train, X_val, y_bin_val,
        etf_names=etf_names, epochs=epochs
    )
    epochs_per = {n: len(h.history['loss']) for n, h in zip(etf_names, histories)}
    return models, epochs_per

def compute_strategy_metrics(models, X_test, y_bin_test, y_fwd_test, test_dates, target_etfs, sofr):
    from models import predict_binary_tfts
    from strategy import execute_strategy, calculate_metrics

    proba = predict_binary_tfts(models, X_test)

    # Accuracy per ETF
    acc_per_etf = {}
    for j, name in enumerate(target_etfs):
        preds_j = (proba[:, j] > 0.5).astype(int)
        acc_per_etf[name] = round(float(np.mean(preds_j == y_bin_test[:, j])), 4)

    # Strategy replay with fixed params
    (strat_rets, _, _, _, _, _, _) = execute_strategy(
        proba, y_fwd_test, test_dates, target_etfs,
        fee_bps=15,
        stop_loss_pct=-0.12,
        z_reentry=1.0,
        sofr=sofr,
        z_min_entry=0.5,
        daily_ret_override=None,
    )
    strat_metrics = calculate_metrics(strat_rets, sofr)
    ann_return_val = round(float(strat_metrics['ann_return']), 6)
    sharpe_val     = round(float(strat_metrics['sharpe']),     6)
    max_dd_val     = round(float(strat_metrics['max_dd']),     6)

    return proba, acc_per_etf, ann_return_val, sharpe_val, max_dd_val, strat_rets

def save_global_model(models, scaler, lookback, lookback_results, target_etfs, input_features, option, token):
    import tensorflow as tf
    import pickle
    import tempfile
    from huggingface_hub import HfApi
    import os

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Build the local folder structure: option_{option}/global_model/
        local_root = os.path.join(tmpdir, f"option_{option}", "global_model")
        os.makedirs(local_root, exist_ok=True)

        # Save each model as .h5
        for etf, model in zip(target_etfs, models):
            model_path = os.path.join(local_root, f"{etf}.h5")
            model.save(model_path)  # Keras 3 uses extension to determine format

        # Save scaler
        scaler_path = os.path.join(local_root, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # Save meta
        meta = {
            "lookback": lookback,
            "lookback_results": lookback_results,
            "target_etfs": target_etfs,
            "input_features": input_features,
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = os.path.join(local_root, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Upload the whole folder in a single commit
        api = HfApi()
        api.upload_folder(
            repo_id=HF_OUTPUT_REPO,
            repo_type="dataset",
            folder_path=os.path.join(tmpdir, f"option_{option}"),
            path_in_repo="",   # upload to root of the dataset repo
            commit_message=f"Global model {option} {datetime.now().strftime('%Y-%m-%d')}",
            token=token,
        )
        log.info(f"✅ Uploaded global model for option {option} in a single commit")

def load_global_model(option, token):
    import tensorflow as tf
    import pickle
    from huggingface_hub import hf_hub_download
    # Download meta first to know ETFs and lookback
    meta_path = download_file_from_hf_dataset(f"option_{option}/global_model/meta.json", token)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    target_etfs = meta['target_etfs']
    lookback = meta['lookback']

    # Download and load each model
    models = []
    for etf in target_etfs:
        model_path = download_file_from_hf_dataset(f"option_{option}/global_model/{etf}.h5", token)
        model = tf.keras.models.load_model(model_path)
        models.append(model)

    # Download scaler
    scaler_path = download_file_from_hf_dataset(f"option_{option}/global_model/scaler.pkl", token)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return models, scaler, meta

# ── Main functions for each mode ──────────────────────────────────────────────
def train_global(option, force_refresh, token):
    """Train a single global model on full history (start_year=2008)."""
    from sklearn.preprocessing import RobustScaler
    from models import SEED, build_binary_tft
    import tensorflow as tf

    log.info(f"Global training for Option {option.upper()}")
    # Prepare data from 2008
    df, df_model, fwd_model, binary_targets, target_etfs, input_features, sofr, rf_label = prepare_data(
        option, start_year=2008, force_refresh=force_refresh)

    # Scale features
    scaler = RobustScaler()
    scaler.fit(df_model[input_features].values)
    scaled = scaler.transform(df_model[input_features].values)

    # Determine optimal lookback
    bin_proxy = binary_targets[target_etfs[0]].values
    lookback_candidates = [20, 30, 40, 50, 60]
    best_lookback, lookback_results = optimize_lookback(scaled, bin_proxy, lookback_candidates, SEED)

    # Create sequences
    bin_vals = binary_targets[target_etfs].values
    fwd_vals = fwd_model[target_etfs].values
    dates = df_model.index
    X_train, y_bin_train, X_val, y_bin_val, X_test, y_fwd_test, y_bin_test, test_dates = create_sequences(
        scaled, bin_vals, fwd_vals, dates, best_lookback, SEED)

    etf_names = [e.replace('_Ret', '') for e in target_etfs]

    # Train models (one per ETF)
    models, epochs_per = train_models(X_train, y_bin_train, X_val, y_bin_val, etf_names, epochs=150, seed=SEED)

    # Compute metrics (for info)
    proba, acc_per_etf, ann_return, sharpe, max_dd, strat_rets = compute_strategy_metrics(
        models, X_test, y_bin_test, y_fwd_test, test_dates, etf_names, sofr)

    log.info(f"Global model metrics: AnnReturn={ann_return*100:.2f}%, Sharpe={sharpe:.2f}, MaxDD={max_dd*100:.2f}%")

    # Save model and scaler to HF
    save_global_model(models, scaler, best_lookback, lookback_results, etf_names, input_features, option, token)

    # Also save a signals.json for the latest (maybe useful)
    last_scores = proba[-1]
    best_idx = np.argmax(last_scores)
    next_signal = etf_names[best_idx]
    from utils import get_next_trading_day
    next_date = get_next_trading_day(test_dates[-1])
    signals_payload = {
        "next_signal": next_signal,
        "next_date": str(next_date),
        "conviction_z": float((last_scores[best_idx] - 0.5) * 2),
        "conviction_label": "High" if last_scores[best_idx] > 0.7 else "Medium" if last_scores[best_idx] > 0.55 else "Low",
        "etf_scores": {name: float(score) for name, score in zip(etf_names, last_scores)},
        "sofr": sofr,
        "rf_label": rf_label,
        "data_start": str(df.index[0].date()),
        "data_end": str(df.index[-1].date()),
        "test_start": str(test_dates[0].date()),
        "test_end": str(test_dates[-1].date()),
        "n_test_days": len(test_dates),
        "lookback_days": best_lookback,
        "start_year": 2008,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    push_file_to_hf_dataset(
        f"option_{option}/global_model/signals.json",
        json.dumps(signals_payload, indent=2).encode(),
        f"Global signals {datetime.now().strftime('%Y-%m-%d')}",
        token,
    )

    log.info("Global model training complete")

def predict_global(option, year, sweep_date, force_refresh, token):
    """Load global model and generate sweep JSON for a specific year."""
    from models import predict_binary_tfts
    from sklearn.preprocessing import RobustScaler
    import tensorflow as tf
    from strategy import execute_strategy, calculate_metrics
    from utils import get_next_trading_day

    log.info(f"Global prediction for Option {option.upper()}, year {year}")
    # Load global model
    models, scaler, meta = load_global_model(option, token)
    lookback = meta['lookback']
    target_etfs = meta['target_etfs']
    input_features = meta['input_features']

    # Load data only up to the end of the target year
    end_date = pd.Timestamp(f"{year}-12-31")
    df_full, df_model, fwd_model, binary_targets, _, _, sofr, rf_label = prepare_data(
        option, start_year=2008, force_refresh=force_refresh)
    df_model_year = df_model[df_model.index <= end_date]

    if len(df_model_year) == 0:
        log.warning(f"No data in df_model for year {year}")
        return

    # Scale features for the whole df_model (already scaled globally? We'll rescale with the global scaler)
    X_full_scaled = scaler.transform(df_model[input_features].values)

    # Create sequences for the entire df_model
    dates = df_model.index
    X_seq = []
    seq_dates = []
    for i in range(lookback, len(X_full_scaled) - 1):
        X_seq.append(X_full_scaled[i - lookback:i])
        seq_dates.append(dates[i + 1])  # the date of the prediction (T+1)
    X_seq = np.array(X_seq, dtype=np.float32)
    seq_dates = pd.DatetimeIndex(seq_dates)

    # Keep only those in the target year
    mask = (seq_dates >= pd.Timestamp(f"{year}-01-01")) & (seq_dates <= end_date)
    X_test_year = X_seq[mask]
    test_dates_year = seq_dates[mask]

    if len(X_test_year) == 0:
        log.warning(f"No test sequences found for year {year}")
        return

    # Get corresponding y_fwd and y_bin from binary_targets and fwd_model
    y_fwd_full = fwd_model[target_etfs].loc[seq_dates]
    y_bin_full = binary_targets[target_etfs].loc[seq_dates]
    y_fwd_test = y_fwd_full.loc[test_dates_year].values
    y_bin_test = y_bin_full.loc[test_dates_year].values

    # Predict
    proba = predict_binary_tfts(models, X_test_year)

    # Compute strategy metrics for the year
    daily_ret_test = df_full.loc[test_dates_year][target_etfs].fillna(0.0).values

    (strat_rets, _, _, _, _, _, _) = execute_strategy(
        proba, y_fwd_test, test_dates_year, target_etfs,
        fee_bps=15,
        stop_loss_pct=-0.12,
        z_reentry=1.0,
        sofr=sofr,
        z_min_entry=0.5,
        daily_ret_override=daily_ret_test,
    )
    strat_metrics = calculate_metrics(strat_rets, sofr)
    ann_return_val = round(float(strat_metrics['ann_return']), 6)
    sharpe_val     = round(float(strat_metrics['sharpe']),     6)
    max_dd_val     = round(float(strat_metrics['max_dd']),     6)

    # Last signal for this year
    last_scores = proba[-1]
    best_idx = np.argmax(last_scores)
    next_signal = target_etfs[best_idx]
    next_date = get_next_trading_day(test_dates_year[-1])
    conviction_z = float((last_scores[best_idx] - 0.5) * 2)
    conviction_label = "High" if last_scores[best_idx] > 0.7 else "Medium" if last_scores[best_idx] > 0.55 else "Low"

    # Build sweep data
    sweep_data = {
        "next_signal": next_signal,
        "conviction_z": conviction_z,
        "conviction_label": conviction_label,
        "ann_return": ann_return_val,
        "sharpe": sharpe_val,
        "max_dd": max_dd_val,
        "lookback_days": lookback,
        "start_year": year,
        "sweep_date": sweep_date,
        "etf_scores": {name: float(score) for name, score in zip(target_etfs, last_scores)},
        "model_type": "global",
    }

    # Push sweep file
    sweep_subdir = f"global_sweep/option_{option}"
    sweep_fname = f"{sweep_subdir}/signals_{year}_{sweep_date}.json"
    push_file_to_hf_dataset(
        sweep_fname,
        json.dumps(sweep_data, indent=2).encode(),
        f"[global sweep] {year} {sweep_date} → {next_signal}",
        token,
    )
    log.info(f"✅ Global sweep cache pushed: {sweep_fname}  signal={next_signal}  z={conviction_z:.3f}")

def train_year(option, force_refresh, start_year, sweep_date, token):
    """Original per-year training and sweep."""
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import RobustScaler
    from models import SEED, build_binary_tft, train_all_binary_tfts, predict_binary_tfts
    from strategy import execute_strategy, calculate_metrics, compute_signal_conviction
    from utils import get_next_trading_day
    import tensorflow as tf

    df, df_model, fwd_model, binary_targets, target_etfs, input_features, sofr, rf_label = prepare_data(
        option, start_year, force_refresh)

    # Scale features
    scaler = RobustScaler()
    approx_train_end = int((len(df_model) - 60 - 1) * TRAIN_PCT)
    scaler.fit(df_model[input_features].values[:approx_train_end])
    scaled = scaler.transform(df_model[input_features].values)

    # Optimize lookback
    bin_proxy = binary_targets[target_etfs[0]].values
    lookback_candidates = [20, 30, 40, 50, 60]
    best_lookback, lookback_results = optimize_lookback(scaled, bin_proxy, lookback_candidates, SEED)

    # Create sequences
    bin_vals = binary_targets[target_etfs].values
    fwd_vals = fwd_model[target_etfs].values
    dates = df_model.index
    X_train, y_bin_train, X_val, y_bin_val, X_test, y_fwd_test, y_bin_test, test_dates = create_sequences(
        scaled, bin_vals, fwd_vals, dates, best_lookback, SEED)

    etf_names = [e.replace('_Ret', '') for e in target_etfs]

    # Train models
    models, epochs_per = train_models(X_train, y_bin_train, X_val, y_bin_val, etf_names, epochs=150, seed=SEED)

    # Compute metrics
    proba, acc_per_etf, ann_return_val, sharpe_val, max_dd_val, strat_rets = compute_strategy_metrics(
        models, X_test, y_bin_test, y_fwd_test, test_dates, etf_names, sofr)

    # Last signal
    last_scores = proba[-1]
    best_idx, conviction_z, conviction_label = compute_signal_conviction(last_scores)
    next_signal = etf_names[best_idx]
    next_date = get_next_trading_day(test_dates[-1])

    # Build outputs
    signals_payload = {
        "next_signal": next_signal,
        "next_date": str(next_date),
        "conviction_z": round(float(conviction_z), 4),
        "conviction_label": conviction_label,
        "etf_scores": {name: round(float(score), 4) for name, score in zip(etf_names, last_scores)},
        "sofr": round(sofr, 6),
        "rf_label": rf_label,
        "data_start": str(df.index[0].date()),
        "data_end": str(df.index[-1].date()),
        "test_start": str(test_dates[0].date()),
        "test_end": str(test_dates[-1].date()),
        "n_test_days": len(test_dates),
        "lookback_days": best_lookback,
        "start_year": start_year,
        "ann_return": ann_return_val,
        "sharpe": sharpe_val,
        "max_dd": max_dd_val,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # model_outputs.npz
    daily_ret_test = df.reindex(test_dates)[target_etfs].fillna(0.0).values
    spy_ret_test = df.reindex(test_dates)['SPY_Ret'].fillna(0.0).values if 'SPY_Ret' in df.columns else np.zeros(len(test_dates))
    agg_ret_test = df.reindex(test_dates)['AGG_Ret'].fillna(0.0).values if 'AGG_Ret' in df.columns else np.zeros(len(test_dates))
    npz_buf = {
        'proba': proba.astype(np.float32),
        'daily_ret_test': daily_ret_test.astype(np.float32),
        'y_fwd_test': y_fwd_test.astype(np.float32),
        'spy_ret_test': spy_ret_test.astype(np.float32),
        'agg_ret_test': agg_ret_test.astype(np.float32),
        'test_dates': np.array([str(d.date()) for d in test_dates]),
        'target_etfs': np.array(target_etfs),
        'sofr': np.array([sofr]),
        'all_test_start': np.array([str(test_dates[0].date())]),
        'all_test_end': np.array([str(test_dates[-1].date())]),
    }
    npz_io = io.BytesIO()
    np.savez_compressed(npz_io, **npz_buf)
    npz_bytes = npz_io.getvalue()

    # training_meta.json
    meta_payload = {
        "lookback_days": best_lookback,
        "lookback_search": lookback_results,
        "epochs_per_etf": epochs_per,
        "accuracy_per_etf": acc_per_etf,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "n_features": len(input_features),
        "n_targets": len(target_etfs),
        "target_etfs": target_etfs,
        "split": "80/10/10",
        "forward_days": FORWARD_DAYS,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    output_subdir = f"option_{option}"
    # Push main outputs
    push_file_to_hf_dataset(f"{output_subdir}/model_outputs.npz", npz_bytes,
                            f"Update model outputs {datetime.now().strftime('%Y-%m-%d')} start_year={start_year}", token)
    push_file_to_hf_dataset(f"{output_subdir}/signals.json",
                            json.dumps(signals_payload, indent=2).encode(),
                            f"Update signals {datetime.now().strftime('%Y-%m-%d')} → {next_signal}", token)
    push_file_to_hf_dataset(f"{output_subdir}/training_meta.json",
                            json.dumps(meta_payload, indent=2).encode(),
                            f"Update training meta {datetime.now().strftime('%Y-%m-%d')}", token)

    # Sweep file if year is in SWEEP_YEARS (adjust as needed)
    SWEEP_YEARS = [2008, 2014, 2016, 2019, 2021]
    if start_year in SWEEP_YEARS:
        sweep_subdir = f"sweep/option_{option}"
        sweep_fname = f"{sweep_subdir}/signals_{start_year}_{sweep_date}.json"
        sweep_data = {
            "next_signal": next_signal,
            "conviction_z": round(float(conviction_z), 4),
            "conviction_label": conviction_label,
            "ann_return": ann_return_val,
            "sharpe": sharpe_val,
            "max_dd": max_dd_val,
            "lookback_days": best_lookback,
            "start_year": start_year,
            "sweep_date": sweep_date,
            "etf_scores": {name: round(float(score), 4) for name, score in zip(etf_names, last_scores)},
        }
        push_file_to_hf_dataset(sweep_fname, json.dumps(sweep_data, indent=2).encode(),
                                f"[sweep] {start_year} {sweep_date} → {next_signal}", token)
        log.info(f"✅ Sweep cache pushed: {sweep_fname}  signal={next_signal}")

    log.info(f"Per-year training complete for option {option}, start_year={start_year}")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["a", "b"], default="a",
                        help="Which ETF universe to train: a (FI/Commodities) or b (Equity)")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force full dataset rebuild")
    parser.add_argument("--start-year", type=int, default=None,
                        help="Override training start year (for per-year mode)")
    parser.add_argument("--sweep-date", default=None,
                        help="Date tag for sweep cache file (YYYYMMDD)")
    parser.add_argument("--mode", choices=["train-year", "train-global", "predict-global"],
                        default="train-year", help="Mode of operation")
    parser.add_argument("--year", type=int, default=None,
                        help="Year for prediction (predict-global mode)")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable not set")

    if args.mode == "train-global":
        train_global(args.option, args.force_refresh, token)
    elif args.mode == "predict-global":
        if args.year is None:
            raise ValueError("--year required for predict-global mode")
        sweep_date = args.sweep_date or (datetime.now(timezone.utc) - timedelta(hours=5)).strftime("%Y%m%d")
        predict_global(args.option, args.year, sweep_date, args.force_refresh, token)
    else:  # train-year
        if args.start_year is None:
            args.start_year = 2008
        sweep_date = args.sweep_date or (datetime.now(timezone.utc) - timedelta(hours=5)).strftime("%Y%m%d")
        train_year(args.option, args.force_refresh, args.start_year, sweep_date, token)
