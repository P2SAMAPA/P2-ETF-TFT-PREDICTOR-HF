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
    """
    Execute strategy on test set.
    NOTE: We do NOT filter by NYSE calendar here — test_dates already come
    from the dataset which only contains trading days. Calendar filtering
    was dropping recent dates due to calendar library lag.
    """
    # Convert to pandas DatetimeIndex if needed
    if not isinstance(test_dates, pd.DatetimeIndex):
        test_dates = pd.DatetimeIndex(test_dates)

    # Convert arrays to numpy
    proba      = np.array(proba)
    y_fwd_test = np.array(y_fwd_test)
    if daily_ret_override is not None:
        daily_ret_override = np.array(daily_ret_override)

    strat_rets      = []
    audit_trail     = []
    daily_rf        = sofr / 252
    stop_active     = False
    recent_rets     = []   # last 2 net returns for stop-loss check
    top_pick_rets   = []   # last 5 TOP-PICK actual daily returns for rotation
    rotated_etf_idx = None
    today           = datetime.now().date()

    for i in range(len(proba)):
        day_scores     = np.array(proba[i], dtype=float)
        ranked_indices = np.argsort(day_scores)[::-1]
        best_idx       = int(ranked_indices[0])
        second_idx     = int(ranked_indices[1]) if len(ranked_indices) > 1 else best_idx

        # Top pick's actual daily return (always track #1 pick for rotation)
        top_actual = (float(daily_ret_override[i][best_idx])
                      if daily_ret_override is not None
                      else float(y_fwd_test[i][best_idx]))

        # ── 5-day consecutive loss rotation ──────────────────────────────────
        # Buffer contains PREVIOUS days' returns (appended at end of loop)
        # So on day i, top_pick_rets has returns from days [i-5 .. i-1]
        # Check: all 5 previous days negative → rotate to #2
        # Recovery: top pick positive today → rotate back to #1
        if rotated_etf_idx is not None:
            # Currently rotated — check if top pick has recovered today
            if top_actual > 0:
                rotated_etf_idx = None
        
        # Only check rotation trigger if not already rotated
        if rotated_etf_idx is None and len(top_pick_rets) >= 5:
            if all(r < 0 for r in top_pick_rets[-5:]):
                rotated_etf_idx = second_idx

        active_idx  = rotated_etf_idx if rotated_etf_idx is not None else best_idx
        _, day_z, _ = compute_signal_conviction(day_scores)
        signal_etf  = target_etfs[active_idx].replace('_Ret', '')

        realized_ret = (float(daily_ret_override[i][active_idx])
                        if daily_ret_override is not None
                        else float(y_fwd_test[i][active_idx]))

        # ── Stop-loss + conviction gate ───────────────────────────────────────
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

        # ── Update rolling buffers AFTER decision ────────────────────────────
        recent_rets.append(net_ret)
        if len(recent_rets) > 2:
            recent_rets.pop(0)

        top_pick_rets.append(top_actual)
        if len(top_pick_rets) > 5:
            top_pick_rets.pop(0)

        # ── Audit trail — all dates up to and including today ─────────────────
        trade_date = test_dates[i]
        if hasattr(trade_date, 'date'):
            trade_date_val = trade_date.date()
        else:
            trade_date_val = trade_date

        if trade_date_val <= today:
            audit_trail.append({
                'Date':         trade_date_val.strftime('%Y-%m-%d'),
                'Signal':       trade_signal,
                'Conviction_Z': round(day_z, 2),
                'Net_Return':   round(net_ret, 5),
                'Stop_Active':  stop_active,
                'Rotated':      rotated_etf_idx is not None
            })

    strat_rets = np.array(strat_rets)

    # ── Next trading day signal ───────────────────────────────────────────────
    if len(test_dates) > 0 and len(proba) > 0:
        last_date         = test_dates[-1]
        next_trading_date = get_next_trading_day(last_date)
        last_scores       = np.array(proba[-1], dtype=float)
        next_best_idx, conviction_zscore, conviction_label = \
            compute_signal_conviction(last_scores)
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
