"""
Backtesting and strategy execution logic — TFT model only.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from utils import filter_to_trading_days, get_next_trading_day


def compute_signal_conviction(raw_scores):
    best_idx = int(np.argmax(raw_scores))
    mean = np.mean(raw_scores)
    std  = np.std(raw_scores)
    z = 0.0 if std < 1e-9 else (raw_scores[best_idx] - mean) / std
    if   z >= 2.0: label = "Very High"
    elif z >= 1.0: label = "High"
    elif z >= 0.0: label = "Moderate"
    else:          label = "Low"
    return best_idx, z, label


def execute_strategy(proba, y_fwd_test, test_dates, target_etfs, fee_bps,
                     stop_loss_pct=-0.12, z_reentry=1.0,
                     sofr=0.045, z_min_entry=0.5,
                     daily_ret_override=None):
    arrays_to_filter = [proba, y_fwd_test]
    if daily_ret_override is not None:
        arrays_to_filter.append(daily_ret_override)

    filtered_dates, filtered_arrays = filter_to_trading_days(test_dates, arrays_to_filter)
    test_dates = filtered_dates
    proba      = filtered_arrays[0]
    y_fwd_test = filtered_arrays[1]
    if daily_ret_override is not None:
        daily_ret_override = filtered_arrays[2]

    strat_rets  = []
    audit_trail = []
    daily_rf    = sofr / 252
    stop_active = False
    recent_rets = []
    today       = datetime.now().date()

    for i in range(len(proba)):
        day_scores = np.array(proba[i], dtype=float)
        best_idx, day_z, _ = compute_signal_conviction(day_scores)
        signal_etf = target_etfs[best_idx].replace('_Ret', '')

        if daily_ret_override is not None:
            realized_ret = float(daily_ret_override[i][best_idx])
        else:
            realized_ret = float(y_fwd_test[i][best_idx])

        if stop_active:
            if day_z >= z_reentry and day_z >= z_min_entry:
                stop_active  = False
                net_ret      = realized_ret - (fee_bps / 10000)
                trade_signal = signal_etf
            else:
                net_ret      = daily_rf
                trade_signal = "CASH"
        else:
            if day_z < z_min_entry:
                net_ret      = daily_rf
                trade_signal = "CASH"
            else:
                if len(recent_rets) >= 2:
                    cum_2d = (1 + recent_rets[-2]) * (1 + recent_rets[-1]) - 1
                    if cum_2d <= stop_loss_pct:
                        stop_active  = True
                        net_ret      = daily_rf
                        trade_signal = "CASH"
                    else:
                        net_ret      = realized_ret - (fee_bps / 10000)
                        trade_signal = signal_etf
                else:
                    net_ret      = realized_ret - (fee_bps / 10000)
                    trade_signal = signal_etf

        strat_rets.append(net_ret)
        recent_rets.append(net_ret)
        if len(recent_rets) > 2:
            recent_rets.pop(0)

        trade_date = test_dates[i]
        if trade_date.date() < today:
            audit_trail.append({
                'Date':         trade_date.strftime('%Y-%m-%d'),
                'Signal':       trade_signal,
                'Conviction_Z': round(day_z, 2),
                'Realized':     round(realized_ret, 5),
                'Net_Return':   round(net_ret, 5),
                'Stop_Active':  stop_active
            })

    strat_rets = np.array(strat_rets)

    if len(test_dates) > 0 and len(proba) > 0:
        last_date         = test_dates[-1]
        next_trading_date = get_next_trading_day(last_date)
        last_scores       = np.array(proba[-1], dtype=float)
        next_best_idx, conviction_zscore, conviction_label = compute_signal_conviction(last_scores)
        next_signal    = target_etfs[next_best_idx].replace('_Ret', '')
        all_etf_scores = last_scores
    else:
        next_trading_date = datetime.now().date()
        next_signal       = "CASH"
        conviction_zscore = 0.0
        conviction_label  = "Low"
        all_etf_scores    = np.zeros(len(target_etfs))

    return (strat_rets, audit_trail, next_signal, next_trading_date,
            conviction_zscore, conviction_label, all_etf_scores)


def calculate_metrics(strat_rets, sofr_rate=0.045):
    cum_returns  = np.cumprod(1 + strat_rets)
    ann_return   = (cum_returns[-1] ** (252 / len(strat_rets))) - 1
    sharpe       = ((np.mean(strat_rets) - sofr_rate / 252) /
                    (np.std(strat_rets) + 1e-9) * np.sqrt(252))
    recent_rets  = strat_rets[-15:]
    hit_ratio    = np.mean(recent_rets > 0)
    cum_max      = np.maximum.accumulate(cum_returns)
    drawdown     = (cum_returns - cum_max) / cum_max
    max_dd       = np.min(drawdown)
    max_daily_dd = np.min(strat_rets)
    return {
        'cum_returns':  cum_returns,
        'ann_return':   ann_return,
        'sharpe':       sharpe,
        'hit_ratio':    hit_ratio,
        'max_dd':       max_dd,
        'max_daily_dd': max_daily_dd,
        'cum_max':      cum_max
    }


def calculate_benchmark_metrics(benchmark_returns, sofr_rate=0.045):
    cum_returns  = np.cumprod(1 + benchmark_returns)
    ann_return   = (cum_returns[-1] ** (252 / len(benchmark_returns))) - 1
    sharpe       = ((np.mean(benchmark_returns) - sofr_rate / 252) /
                    (np.std(benchmark_returns) + 1e-9) * np.sqrt(252))
    cum_max      = np.maximum.accumulate(cum_returns)
    dd           = (cum_returns - cum_max) / cum_max
    max_dd       = np.min(dd)
    max_daily_dd = np.min(benchmark_returns)
    return {
        'cum_returns':  cum_returns,
        'ann_return':   ann_return,
        'sharpe':       sharpe,
        'max_dd':       max_dd,
        'max_daily_dd': max_daily_dd
    }
