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
                     model_type="ensemble",
                     stop_loss_pct=-0.12, z_reentry=1.0,
                     sofr=0.045, all_proba=None, z_min_entry=0.5):
    """
    Execute trading strategy with T+1 execution, trailing stop-loss,
    and minimum conviction gate.

    Stop-loss    : if 2-day cumulative return ≤ stop_loss_pct → CASH earning Rf
    Re-entry     : return to ETF when conviction Z-score ≥ z_reentry
    Conviction gate: only enter ETF if conviction Z-score ≥ z_min_entry,
                     otherwise hold CASH (avoids low-confidence trades)
    """

    # Filter to only trading days
    if model_type == "ensemble":
        filtered_dates, filtered_data = filter_to_trading_days(
            test_dates, [preds, y_raw_test]
        )
        preds, y_raw_test = filtered_data
        if all_proba is not None:
            _, [all_proba] = filter_to_trading_days(test_dates, [all_proba])
    else:
        filtered_dates, filtered_data = filter_to_trading_days(
            test_dates, [preds, y_raw_test]
        )
        preds, y_test = filtered_data

    test_dates = filtered_dates

    strat_rets = []
    audit_trail = []
    daily_rf = sofr / 252          # daily risk-free rate earned while in CASH

    stop_active = False            # True = stop triggered, holding CASH
    recent_rets = []               # rolling buffer for 2-day cumulative return check

    num_realized = len(preds)
    today = datetime.now().date()

    for i in range(num_realized):
        # ── Get model scores for conviction Z-score ──────────────────────────
        if model_type == "ensemble":
            best_idx = int(preds[i])
            signal_etf = target_etfs[best_idx].replace('_Ret', '')
            realized_ret = y_raw_test[i][best_idx]
            # Use full per-day probabilities if available, else one-hot
            if all_proba is not None:
                day_scores = np.array(all_proba[i], dtype=float)
            else:
                day_scores = np.zeros(len(target_etfs))
                day_scores[best_idx] = 1.0
        else:
            best_idx = np.argmax(preds[i])
            signal_etf = target_etfs[best_idx].replace('_Ret', '')
            realized_ret = y_test[i][best_idx]
            day_scores = np.array(preds[i], dtype=float)

        # ── Conviction Z-score for today ─────────────────────────────────────
        _, day_z, _ = compute_signal_conviction(day_scores)

        # ── Stop-loss logic ──────────────────────────────────────────────────
        if stop_active:
            # Stay in CASH until conviction Z-score exceeds re-entry threshold
            if day_z >= z_reentry:
                stop_active = False
                # Re-enter only if conviction also meets minimum entry bar
                if day_z >= z_min_entry:
                    net_ret = realized_ret - (fee_bps / 10000)
                    trade_signal = signal_etf
                else:
                    net_ret = daily_rf
                    trade_signal = "CASH"
            else:
                # Remain in CASH — earn daily Rf, no fee
                net_ret = daily_rf
                trade_signal = "CASH"
        else:
            # ── Conviction gate: only trade if model is decisive enough ──────
            if day_z < z_min_entry:
                net_ret = daily_rf
                trade_signal = "CASH"
            else:
                # Check 2-day cumulative return for stop trigger
                if len(recent_rets) >= 2:
                    cum_2d = (1 + recent_rets[-2]) * (1 + recent_rets[-1]) - 1
                    if cum_2d <= stop_loss_pct:
                        stop_active = True
                        net_ret = daily_rf      # switch to CASH immediately today
                        trade_signal = "CASH"
                    else:
                        net_ret = realized_ret - (fee_bps / 10000)
                        trade_signal = signal_etf
                else:
                    net_ret = realized_ret - (fee_bps / 10000)
                    trade_signal = signal_etf

        strat_rets.append(net_ret)
        recent_rets.append(net_ret)
        if len(recent_rets) > 2:
            recent_rets.pop(0)

        trade_date = test_dates[i]
        if trade_date.date() < today:
            audit_trail.append({
                'Date': trade_date.strftime('%Y-%m-%d'),
                'Signal': trade_signal,
                'Realized': realized_ret,
                'Net_Return': net_ret,
                'Stop_Active': stop_active
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
