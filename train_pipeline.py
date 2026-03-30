"""
train_pipeline.py — FINAL CORRECTED VERSION with separate option folders
Global model training writes to global_model/option_{a,b}/
"""

import os, sys, json, logging, argparse, numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
import io, pickle, random, tempfile, time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = "42"

HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-tft-outputs"
HF_DATASET_REPO = "P2SAMAPA/my-etf-data"

TRAIN_PCT, VAL_PCT, TEST_PCT, FORWARD_DAYS = 0.80, 0.10, 0.10, 5


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT MOCK
# ─────────────────────────────────────────────────────────────────────────────
def _make_st_mock():
    import unittest.mock as mock
    st = mock.MagicMock()
    st.warning = lambda *a, **k: log.warning(" ".join(map(str, a)))
    st.error   = lambda *a, **k: log.error(" ".join(map(str, a)))
    st.info    = lambda *a, **k: log.info(" ".join(map(str, a)))
    st.success = lambda *a, **k: log.info(" ".join(map(str, a)))
    st.write   = lambda *a, **k: log.info(" ".join(map(str, a)))
    cm = mock.MagicMock()
    cm.__enter__ = lambda s: s
    cm.__exit__  = mock.MagicMock(return_value=False)
    st.status  = mock.MagicMock(return_value=cm)
    st.spinner = mock.MagicMock(return_value=cm)
    st.secrets = {}
    return st

sys.modules["streamlit"] = _make_st_mock()


# ─────────────────────────────────────────────────────────────────────────────
# HF UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def push_file_to_hf_dataset(filename, content_bytes, commit_msg, token, max_retries=3):
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    for attempt in range(max_retries):
        try:
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                tmp.write(content_bytes)
                tmp.flush()

                HfApi().upload_file(
                    path_or_fileobj=tmp.name,
                    path_in_repo=filename,
                    repo_id=HF_OUTPUT_REPO,
                    repo_type="dataset",
                    token=token,
                    commit_message=commit_msg,
                )

            log.info(f"✅ Uploaded {filename}")
            return

        except HfHubHTTPError as e:
            if e.response.status_code in (412, 429):
                wait = 2 ** attempt
                log.warning(f"Retry {attempt+1}/{max_retries} in {wait}s")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Upload failed: {filename}")


def download_file_from_hf_dataset(filename, token):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=HF_OUTPUT_REPO,
        repo_type="dataset",
        filename=filename,
        token=token,
    )
    log.info(f"Downloaded {filename}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
def fetch_sofr():
    try:
        import pandas_datareader.data as web
        dtb3 = web.DataReader('DTB3', 'fred', start='2024-01-01').dropna()
        if not dtb3.empty:
            rate = float(dtb3.iloc[-1].values[0]) / 100
            return rate, f"FRED DTB3 {dtb3.index[-1].date()}"
    except Exception as e:
        log.warning(f"FRED fetch failed: {e}")
    return 0.045, "fallback 4.5%"


def prepare_data(option, start_year, force_refresh):
    from config import OPTION_A_ETFS, OPTION_B_ETFS
    from data_manager import get_data

    TARGET_ETF_LABELS = OPTION_A_ETFS if option == 'a' else OPTION_B_ETFS
    df = get_data(start_year=start_year, force_refresh=force_refresh, clean_hf_dataset=False)

    target_etfs = [c for c in df.columns if c.endswith('_Ret') and any(e in c for e in TARGET_ETF_LABELS)]

    input_features = [
        c for c in df.columns
        if (c.endswith('_Z') or c.endswith('_Vol') or 'Regime' in c or 'YC_' in c or
            'Credit_' in c or 'Rates_' in c or 'VIX_Term_' in c or
            'Rising' in c or 'Falling' in c or 'Accelerating' in c)
        and c not in target_etfs
    ]

    sofr, rf_label = fetch_sofr()

    if 'DTB3' in df.columns and 'fallback' in rf_label:
        sofr = float(df['DTB3'].dropna().iloc[-1]) / 100
        rf_label = "dataset DTB3"

    daily_rf_5d = (sofr / 252) * FORWARD_DAYS

    fwd_returns = pd.DataFrame(index=df.index)
    for col in target_etfs:
        fwd_returns[col] = df[col].rolling(FORWARD_DAYS).sum().shift(-FORWARD_DAYS)

    valid_idx = fwd_returns.dropna().index

    df_model = df.loc[valid_idx]
    fwd_model = fwd_returns.loc[valid_idx]

    binary_targets = pd.DataFrame(index=df_model.index)
    for col in target_etfs:
        binary_targets[col] = (fwd_model[col] > daily_rf_5d).astype(np.int32)

    return df, df_model, fwd_model, binary_targets, target_etfs, input_features, sofr, rf_label


# ─────────────────────────────────────────────────────────────────────────────
# 🔴 CRITICAL FIX: Separate folders per option
# ─────────────────────────────────────────────────────────────────────────────
def load_global_model(option, token):
    import tensorflow as tf, pickle
    from models import build_binary_tft

    # FIXED PATH: Now uses global_model/option_{a,b}/ structure
    base = f"global_model/option_{option}"

    meta_path = download_file_from_hf_dataset(f"{base}/meta.json", token)

    with open(meta_path) as f:
        meta = json.load(f)

    lookback = meta['lookback']
    num_features = meta.get('num_features', len(meta['input_features']))
    target_etfs = meta['target_etfs']

    models = []

    for etf in target_etfs:
        try:
            w_path = download_file_from_hf_dataset(f"{base}/{etf}.weights.h5", token)
            model = build_binary_tft(seq_len=lookback, num_features=num_features)
            model.load_weights(w_path)
        except Exception as e:
            log.warning(f"Failed to load weights for {etf}, trying full model: {e}")
            full_path = download_file_from_hf_dataset(f"{base}/{etf}.h5", token)
            model = tf.keras.models.load_model(full_path)

        models.append(model)

    scaler_path = download_file_from_hf_dataset(f"{base}/scaler.pkl", token)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return models, scaler, meta


def predict_global(option, year, sweep_date, force_refresh, token):
    from models import predict_binary_tfts
    from strategy import execute_strategy, calculate_metrics

    log.info(f"Global prediction for Option {option.upper()}, year {year}")

    # Load global model from option-specific folder
    models, scaler, meta = load_global_model(option, token)

    lookback = meta['lookback']
    target_etfs = meta['target_etfs']
    input_features = meta['input_features']

    # Prepare data
    df_full, df_model, fwd_model, binary_targets, _, _, sofr, rf_label = prepare_data(
        option, 2008, force_refresh
    )

    # Scale full dataset
    X_full_scaled = scaler.transform(df_model[input_features].values)
    dates = df_model.index

    # Build sequences
    X_seq, seq_dates = [], []
    for i in range(lookback, len(X_full_scaled) - 1):
        X_seq.append(X_full_scaled[i - lookback:i])
        seq_dates.append(dates[i + 1])

    X_seq = np.array(X_seq, dtype=np.float32)
    seq_dates = pd.DatetimeIndex(seq_dates)

    # Filter for requested year
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")

    mask = (seq_dates >= start) & (seq_dates <= end)

    X_test_year = X_seq[mask]
    test_dates_year = seq_dates[mask]

    if len(X_test_year) == 0:
        log.warning(f"No test sequences for year {year}")
        return

    # Targets
    y_fwd_full = fwd_model[target_etfs].loc[seq_dates]
    y_bin_full = binary_targets[target_etfs].loc[seq_dates]

    y_fwd_test = y_fwd_full.loc[test_dates_year].values
    y_bin_test = y_bin_full.loc[test_dates_year].values

    # Predictions
    proba = predict_binary_tfts(models, X_test_year)

    # Strategy
    daily_ret_test = df_full.loc[test_dates_year][target_etfs].fillna(0.0).values

    strat_rets, _, _, _, _, _, _ = execute_strategy(
        proba,
        y_fwd_test,
        test_dates_year,
        target_etfs,
        fee_bps=15,
        stop_loss_pct=-0.12,
        z_reentry=1.0,
        sofr=sofr,
        z_min_entry=0.5,
        daily_ret_override=daily_ret_test
    )

    metrics = calculate_metrics(strat_rets, sofr)

    ann_return = round(metrics['ann_return'], 6)
    sharpe = round(metrics['sharpe'], 6)
    max_dd = round(metrics['max_dd'], 6)

    # Final signal
    last_scores = proba[-1]
    best_idx = np.argmax(last_scores)

    next_signal = target_etfs[best_idx]

    sweep_data = {
        "next_signal": next_signal,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "lookback_days": lookback,
        "start_year": year,
        "sweep_date": sweep_date,
        "etf_scores": {
            name: float(score)
            for name, score in zip(target_etfs, last_scores)
        },
        "model_type": "global"
    }

    # Save to global_sweep/option_{option}/
    fname = f"global_sweep/option_{option}/signals_{year}_{sweep_date}.json"

    push_file_to_hf_dataset(
        fname,
        json.dumps(sweep_data, indent=2).encode(),
        f"[global sweep] {year}",
        token
    )

    log.info(f"✅ Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def train_global(option, force_refresh, token):
    """Train global model for given option and upload to global_model/option_{option}/"""
    from models import build_binary_tft
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    
    log.info(f"Training global model for Option {option.upper()}")
    
    # Prepare data
    df_full, df_model, fwd_model, binary_targets, target_etfs, input_features, sofr, rf_label = prepare_data(
        option, 2008, force_refresh
    )
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_model[input_features].values)
    
    # Build sequences
    lookback = 60  # or from config
    X_seq, y_seq = [], []
    
    for i in range(lookback, len(X_scaled)):
        X_seq.append(X_scaled[i-lookback:i])
        y_seq.append(binary_targets.iloc[i].values)
    
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.int32)
    
    # Train/val split
    n = len(X_seq)
    train_end = int(n * TRAIN_PCT)
    val_end = int(n * (TRAIN_PCT + VAL_PCT))
    
    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
    
    # 🔴 CRITICAL: Use option-specific folder
    base_path = f"global_model/option_{option}"
    
    # Build and train models
    models = []
    for i, etf in enumerate(target_etfs):
        log.info(f"Training model for {etf}")
        model = build_binary_tft(seq_len=lookback, num_features=len(input_features))
        
        # 🔴 CRITICAL FIX: Compile the model before training
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        # Train
        model.fit(
            X_train, y_train[:, i],
            validation_data=(X_val, y_val[:, i]),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        models.append(model)
        
        # 🔴 CRITICAL FIX: save_weights requires actual file path with .weights.h5 suffix
        with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as tmp_weights:
            model.save_weights(tmp_weights.name)
            tmp_weights.flush()
            with open(tmp_weights.name, 'rb') as f:
                weights_bytes = f.read()
        
        push_file_to_hf_dataset(
            f"{base_path}/{etf}.weights.h5",
            weights_bytes,
            f"Global model {option} {datetime.now().strftime('%Y-%m-%d')}",
            token
        )
    
    # Save scaler to option-specific folder
    scaler_bytes = io.BytesIO()
    pickle.dump(scaler, scaler_bytes)
    push_file_to_hf_dataset(
        f"{base_path}/scaler.pkl",
        scaler_bytes.getvalue(),
        f"Global model {option} {datetime.now().strftime('%Y-%m-%d')}",
        token
    )
    
    # Save meta to option-specific folder
    meta = {
        'lookback': lookback,
        'num_features': len(input_features),
        'target_etfs': target_etfs,
        'input_features': input_features,
        'option': option,
        'train_date': datetime.now().isoformat()
    }
    
    push_file_to_hf_dataset(
        f"{base_path}/meta.json",
        json.dumps(meta, indent=2).encode(),
        f"Global model {option} {datetime.now().strftime('%Y-%m-%d')}",
        token
    )
    
    log.info(f"✅ Global model training complete for Option {option}")


def train_year(option, force_refresh, start_year, sweep_date, token):
    """Train per-year model (placeholder - implement based on your needs)"""
    log.info(f"Training year model for Option {option.upper()}, starting {start_year}")
    # Implementation depends on your specific per-year training logic
    pass


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--option", choices=["a","b"], default="a")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--sweep-date", default=None)
    parser.add_argument("--mode", choices=["train-year","train-global","predict-global"], default="train-year")
    parser.add_argument("--year", type=int, default=None)

    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN missing")

    log.info("Pipeline start")

    if args.mode == "train-global":
        train_global(args.option, args.force_refresh, token)

    elif args.mode == "predict-global":
        if args.year is None:
            raise ValueError("--year required")

        sweep_date = args.sweep_date or (
            datetime.now(timezone.utc) - timedelta(hours=5)
        ).strftime("%Y%m%d")

        predict_global(args.option, args.year, sweep_date, args.force_refresh, token)

    else:
        if args.start_year is None:
            args.start_year = 2008

        sweep_date = args.sweep_date or (
            datetime.now(timezone.utc) - timedelta(hours=5)
        ).strftime("%Y%m%d")

        train_year(args.option, args.force_refresh, args.start_year, sweep_date, token)
