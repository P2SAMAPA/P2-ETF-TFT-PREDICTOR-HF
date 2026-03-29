"""
train_pipeline.py — Headless training script for GitHub Actions.

Global model training: writes to option_{option}/global_model/
Global consensus sweep: reads from option_{option}/global_model/ and writes to global_sweep/
Per-year sweep: writes to sweep/ (never touches global_model/)
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
import time

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

HF_OUTPUT_REPO  = "P2SAMAPA/p2-etf-tft-outputs"
HF_DATASET_REPO = "P2SAMAPA/my-etf-data"

TRAIN_PCT    = 0.80
VAL_PCT      = 0.10
TEST_PCT     = 0.10
FORWARD_DAYS = 5

# ── Streamlit mock ──────────────────────────────────────────────────────────────
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

# ── HF utilities with retry ───────────────────────────────────────────────────
def push_file_to_hf_dataset(filename: str, content_bytes: bytes,
                            commit_msg: str, token: str, max_retries=3):
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    for attempt in range(max_retries):
        try:
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                tmp.write(content_bytes)
                tmp.flush()
                api = HfApi()
                api.upload_file(
                    path_or_fileobj=tmp.name,
                    path_in_repo=filename,
                    repo_id=HF_OUTPUT_REPO,
                    repo_type="dataset",
                    token=token,
                    commit_message=commit_msg,
                )
            log.info(f"✅ Pushed {filename} → {HF_OUTPUT_REPO} ({len(content_bytes):,} bytes)")
            return
        except HfHubHTTPError as e:
            if e.response.status_code in (412, 429):
                wait = 2 ** attempt
                log.warning(f"Upload failed with {e.response.status_code}, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed to upload {filename} after {max_retries} attempts")

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

# ── Data functions ────────────────────────────────────────────────────────────
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

    sofr, rf_label = fetch_sofr()
    if 'DTB3' in df.columns and 'fallback' in rf_label:
        sofr = float(df['DTB3'].dropna().iloc[-1]) / 100
        rf_label = "dataset DTB3"

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

    acc_per_etf = {}
    for j, name in enumerate(target_etfs):
        preds_j = (proba[:, j] > 0.5).astype(int)
        acc_per_etf[name] = round(float(np.mean(preds_j == y_bin_test[:, j])), 4)

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

# ── Global model functions ────────────────────────────────────────────────────
def save_global_model(models, scaler, lookback, lookback_results, target_etfs, input_features, option, token):
    import tensorflow as tf
    import pickle
    import tempfile
    from huggingface_hub import HfApi
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build correct structure: option_{option}/global_model/
        option_dir = os.path.join(tmpdir, f"option_{option}")
        local_root = os.path.join(option_dir, "global_model")
        os.makedirs(local_root, exist_ok=True)

        # Save weights
        for etf, model in zip(target_etfs, models):
            weights_path = os.path.join(local_root, f"{etf}.weights.h5")
            model.save_weights(weights_path)

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
            "num_features": len(input_features),
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = os.path.join(local_root, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Upload the whole option_{option} folder
        api = HfApi()
        api.upload_folder(
            repo_id=HF_OUTPUT_REPO,
            repo_type="dataset",
            folder_path=option_dir,
            path_in_repo="",
            commit_message=f"Global model {option} {datetime.now().strftime('%Y-%m-%d')}",
            token=token,
        )
        log.info(f"✅ Uploaded global model for option {option} in a single commit")

def load_global_model(option, token):
    import tensorflow as tf
    import pickle
    from huggingface_hub import hf_hub_download
    from models import build_binary_tft

    meta_path = download_file_from_hf_dataset(f"option_{option}/global_model/meta.json", token)
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    lookback = meta['lookback']
    if 'num_features' in meta:
        num_features = meta['num_features']
    else:
        num_features = len(meta['input_features'])
    target_etfs = meta['target_etfs']

    models = []
    for etf in target_etfs:
        try:
            weights_path = download_file_from_hf_dataset(f"option_{option}/global_model/{etf}.weights.h5", token)
            model = build_binary_tft(seq_len=lookback, num_features=num_features)
            model.load_weights(weights_path)
            models.append(model)
        except Exception as e:
            log.warning(f"Could not load weights for {etf}, trying full model .h5: {e}")
            full_model_path = download_file_from_hf_dataset(f"option_{option}/global_model/{etf}.h5", token)
            model = tf.keras.models.load_model(full_model_path)
            models.append(model)

    scaler_path = download_file_from_hf_dataset(f"option_{option}/global_model/scaler.pkl", token)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return models, scaler, meta

def train_global(option, force_refresh, token):
    from sklearn.preprocessing import RobustScaler
    from models import SEED
    import tensorflow as tf

    log.info(f"Global training for Option {option.upper()}")
    df, df_model, fwd_model, binary_targets, target_etfs, input_features, sofr, rf_label = prepare_data(
        option, start_year=2008, force_refresh=force_refresh)

    scaler = RobustScaler()
    scaler.fit(df_model[input_features].values)
    scaled = scaler.transform(df_model[input_features].values)

    bin_proxy = binary_targets[target_etfs[0]].values
    lookback_candidates = [20, 30, 40, 50, 60]
    best_lookback, lookback_results = optimize_lookback(scaled, bin_proxy, lookback_candidates, SEED)

    bin_vals = binary_targets[target_etfs].values
    fwd_vals = fwd_model[target_etfs].values
    dates = df_model.index
    X_train, y_bin_train, X_val, y_bin_val, X_test, y_fwd_test, y_bin_test, test_dates = create_sequences(
        scaled, bin_vals, fwd_vals, dates, best_lookback, SEED)

    etf_names = [e.replace('_Ret', '') for e in target_etfs]

    models, epochs_per = train_models(X_train, y_bin_train, X_val, y_bin_val, etf_names, epochs=150, seed=SEED)

    proba, acc_per_etf, ann_return, sharpe, max_dd, strat_rets = compute_strategy_metrics(
        models, X_test, y_bin_test, y_fwd_test, test_dates, etf_names, sofr)

    log.info(f"Global model metrics: AnnReturn={ann_return*100:.2f}%, Sharpe={sharpe:.2f}, MaxDD={max_dd*100:.2f}%")

    save_global_model(models, scaler, best_lookback, lookback_results, etf_names, input_features, option, token)

    # Optional signals.json for latest state
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
    from models import predict_binary_tfts
    from strategy import execute_strategy, calculate_metrics
    from utils import get_next_trading_day

    log.info(f"Global prediction for Option {option.upper()}, year {year}")
    models, scaler, meta = load_global_model(option, token)
    lookback = meta['lookback']
    target_etfs = meta['target_etfs']
    input_features = meta['input_features']

    end_date = pd.Timestamp(f"{year}-12-31")
    df_full, df_model, fwd_model, binary_targets, _, _, sofr, rf_label = prepare_data(
        option, start_year=2008, force_refresh=force_refresh)
    df_model_year = df_model[df_model.index <= end_date]
    if len(df_model_year) == 0:
        log.warning(f"No data in df_model for year {year}")
        return

    X_full_scaled = scaler.transform(df_model[input_features].values)
    dates = df_model.index
    X_seq = []
    seq_dates = []
    for i in range(lookback, len(X_full_scaled) - 1):
        X_seq.append(X_full_scaled[i - lookback:i])
        seq_dates.append(dates[i + 1])
    X_seq = np.array(X_seq, dtype=np.float32)
    seq_dates = pd.DatetimeIndex(seq_dates)

    mask = (seq_dates >= pd.Timestamp(f"{year}-01-01")) & (seq_dates <= end_date)
    X_test_year = X_seq[mask]
    test_dates_year = seq_dates[mask]
    if len(X_test_year) == 0:
        log.warning(f"No test sequences found for year {year}")
        return

    y_fwd_full = fwd_model[target_etfs].loc[seq_dates]
    y_bin_full = binary_targets[target_etfs].loc[seq_dates]
    y_fwd_test = y_fwd_full.loc[test_dates_year].values
    y_bin_test = y_bin_full.loc[test_dates_year].values

    proba = predict_binary_tfts(models, X_test_year)
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

    last_scores = proba[-1]
    best_idx = np.argmax(last_scores)
    next_signal = target_etfs[best_idx]
    conviction_z = float((last_scores[best_idx] - 0.5) * 2)
    conviction_label = "High" if last_scores[best_idx] > 0.7 else "Medium" if last_scores[best_idx] > 0.55 else "Low"

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
    sweep_subdir = f"global_sweep/option_{option}"
    sweep_fname = f"{sweep_subdir}/signals_{year}_{sweep_date}.json"
    push_file_to_hf_dataset(
        sweep_fname,
        json.dumps(sweep_data, indent=2).encode(),
        f"[global sweep] {year} {sweep_date} → {next_signal}",
        token,
    )
    log.info(f"✅ Global sweep cache pushed: {sweep_fname}")

def train_year(option, force_refresh, start_year, sweep_date, token):
    from sklearn.preprocessing import RobustScaler
    from models import SEED
    from strategy import compute_signal_conviction
    from utils import get_next_trading_day
    import tensorflow as tf

    df, df_model, fwd_model, binary_targets, target_etfs, input_features, sofr, rf_label = prepare_data(
        option, start_year, force_refresh)

    scaler = RobustScaler()
    approx_train_end = int((len(df_model) - 60 - 1) * TRAIN_PCT)
    scaler.fit(df_model[input_features].values[:approx_train_end])
    scaled = scaler.transform(df_model[input_features].values)

    bin_proxy = binary_targets[target_etfs[0]].values
    lookback_candidates = [20, 30, 40, 50, 60]
    best_lookback, lookback_results = optimize_lookback(scaled, bin_proxy, lookback_candidates, SEED)

    bin_vals = binary_targets[target_etfs].values
    fwd_vals = fwd_model[target_etfs].values
    dates = df_model.index
    X_train, y_bin_train, X_val, y_bin_val, X_test, y_fwd_test, y_bin_test, test_dates = create_sequences(
        scaled, bin_vals, fwd_vals, dates, best_lookback, SEED)

    etf_names = [e.replace('_Ret', '') for e in target_etfs]

    models, epochs_per = train_models(X_train, y_bin_train, X_val, y_bin_val, etf_names, epochs=150, seed=SEED)

    proba, acc_per_etf, ann_return_val, sharpe_val, max_dd_val, strat_rets = compute_strategy_metrics(
        models, X_test, y_bin_test, y_fwd_test, test_dates, etf_names, sofr)

    last_scores = proba[-1]
    best_idx, conviction_z, conviction_label = compute_signal_conviction(last_scores)
    next_signal = etf_names[best_idx]

    # Only write the sweep JSON
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
    push_file_to_hf_dataset(
        sweep_fname,
        json.dumps(sweep_data, indent=2).encode(),
        f"[sweep] {start_year} {sweep_date} → {next_signal}",
        token,
    )
    log.info(f"✅ Sweep cache pushed: {sweep_fname}")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["a", "b"], default="a")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--sweep-date", default=None)
    parser.add_argument("--mode", choices=["train-year", "train-global", "predict-global"], default="train-year")
    parser.add_argument("--year", type=int, default=None)
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
