"""
Backtesting and strategy execution logic
"""

import numpy as np
import pandas as pd
from datetime import datetime
from utils import filter_to_trading_days, get_next_trading_day


def compute_signal_conviction(raw_scores):
    """
    Compute Z-score conviction for the selected ETF signal.

    Args:
        raw_scores: 1-D numpy array of model scores/probabilities for each ETF
                    (e.g. class probabilities from RF/XGB, or raw return preds)

    Returns:
        best_idx   : index of the chosen ETF
        z_score    : z-score of the best ETF score vs the distribution of all scores
        conviction : human-readable label  (Very High / High / Moderate / Low)
    """
    best_idx = int(np.argmax(raw_scores))
    mean = np.mean(raw_scores)
    std = np.std(raw_scores)

    if std < 1e-9:
        z = 0.0
    else:
        z = (raw_scores[best_idx] - mean) / std

    if z >= 2.0:
        label = "Very High"
    elif z >= 1.0:
        label = "High"
    elif z >= 0.0:
        label = "Moderate"
    else:
        label = "Low"

    return best_idx, z, label


def execute_strategy(preds, y_raw_test, test_dates, target_etfs, fee_bps,
                     model_type="ensemble"):
    """
    Execute trading strategy with T+1 execution.

    Returns:
        strat_rets        : Strategy returns (numpy array)
        audit_trail       : List of dicts with trade details
        next_signal       : Next trading day's ETF signal (str)
        next_trading_date : Next trading date (date)
        conviction_zscore : Z-score of the next signal vs all ETF scores (float)
        conviction_label  : Human-readable conviction label (str)
        all_etf_scores    : Raw model scores for all ETFs (numpy array) — for UI bar chart
    """

    # Filter to only trading days
    if model_type == "ensemble":
        filtered_dates, filtered_data = filter_to_trading_days(
            test_dates, [preds, y_raw_test]
        )
        preds, y_raw_test = filtered_data
    else:
        filtered_dates, filtered_data = filter_to_trading_days(
            test_dates, [preds, y_raw_test]
        )
        preds, y_test = filtered_data

    test_dates = filtered_dates

    strat_rets = []
    audit_trail = []

    num_realized = len(preds)
    today = datetime.now().date()

    for i in range(num_realized):
        if model_type == "ensemble":
            best_idx = preds[i]
            signal_etf = target_etfs[best_idx].replace('_Ret', '')
            realized_ret = y_raw_test[i][best_idx]
        else:
            best_idx = np.argmax(preds[i])
            signal_etf = target_etfs[best_idx].replace('_Ret', '')
            realized_ret = y_test[i][best_idx]

        net_ret = realized_ret - (fee_bps / 10000)
        strat_rets.append(net_ret)

        trade_date = test_dates[i]
        if trade_date.date() < today:
            audit_trail.append({
                'Date': trade_date.strftime('%Y-%m-%d'),
                'Signal': signal_etf,
                'Realized': realized_ret,
                'Net_Return': net_ret
            })

    strat_rets = np.array(strat_rets)

    # ── Next signal with conviction ──────────────────────────────────────────
    if len(test_dates) > 0:
        last_date = test_dates[-1]
        next_trading_date = get_next_trading_day(last_date)

        if len(preds) > 0:
            last_pred = preds[-1]

            if model_type == "ensemble":
                # Ensemble preds are class indices — we can't get per-ETF scores
                # directly from a single integer.  Use one-hot as a proxy so
                # conviction always reflects "model chose this one class".
                # If you later expose predict_proba, pass that here instead.
                scores = np.zeros(len(target_etfs))
                scores[int(last_pred)] = 1.0
            else:
                scores = np.array(last_pred, dtype=float)

            next_best_idx, conviction_zscore, conviction_label = compute_signal_conviction(scores)
            next_signal = target_etfs[next_best_idx].replace('_Ret', '')
            all_etf_scores = scores
        else:
            next_signal = "CASH"
            conviction_zscore = 0.0
            conviction_label = "Low"
            all_etf_scores = np.zeros(len(target_etfs))
    else:
        next_trading_date = datetime.now().date()
        next_signal = "CASH"
        conviction_zscore = 0.0
        conviction_label = "Low"
        all_etf_scores = np.zeros(len(target_etfs))

    return (strat_rets, audit_trail, next_signal, next_trading_date,
            conviction_zscore, conviction_label, all_etf_scores)


def calculate_metrics(strat_rets, sofr_rate=0.045):
    """Calculate strategy performance metrics"""
    cum_returns = np.cumprod(1 + strat_rets)
    ann_return = (cum_returns[-1] ** (252 / len(strat_rets))) - 1
    sharpe = (np.mean(strat_rets) - (sofr_rate / 252)) / (np.std(strat_rets) + 1e-9) * np.sqrt(252)

    recent_rets = strat_rets[-15:]
    hit_ratio = np.mean(recent_rets > 0)

    cum_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - cum_max) / cum_max
    max_dd = np.min(drawdown)

    max_daily_dd = np.min(strat_rets)

    return {
        'cum_returns': cum_returns,
        'ann_return': ann_return,
        'sharpe': sharpe,
        'hit_ratio': hit_ratio,
        'max_dd': max_dd,
        'max_daily_dd': max_daily_dd,
        'cum_max': cum_max
    }


def calculate_benchmark_metrics(benchmark_returns, sofr_rate=0.045):
    """Calculate benchmark performance metrics"""
    cum_returns = np.cumprod(1 + benchmark_returns)
    ann_return = (cum_returns[-1] ** (252 / len(benchmark_returns))) - 1
    sharpe = (np.mean(benchmark_returns) - (sofr_rate / 252)) / (np.std(benchmark_returns) + 1e-9) * np.sqrt(252)

    cum_max = np.maximum.accumulate(cum_returns)
    dd = (cum_returns - cum_max) / cum_max
    max_dd = np.min(dd)

    max_daily_dd = np.min(benchmark_returns)

    return {
        'cum_returns': cum_returns,
        'ann_return': ann_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'max_daily_dd': max_daily_dd
    }
