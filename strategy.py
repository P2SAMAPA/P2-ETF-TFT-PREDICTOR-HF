"""
Backtesting and strategy execution logic
"""

import numpy as np
import pandas as pd
from utils import filter_to_trading_days, get_next_trading_day


def execute_strategy(preds, y_raw_test, test_dates, target_etfs, fee_bps, 
                     model_type="ensemble"):
    """
    Execute trading strategy with T+1 execution
    
    Args:
        preds: Model predictions (numpy array)
        y_raw_test: Actual returns (numpy array)
        test_dates: DatetimeIndex of test dates
        target_etfs: List of ETF column names
        fee_bps: Transaction fee in basis points
        model_type: "ensemble" or "transformer"
    
    Returns:
        strat_rets: Strategy returns (numpy array)
        audit_trail: List of dicts with trade details
        next_signal: Next trading day's signal
        next_trading_date: Next trading date
    """
    
    # Filter to only trading days
    if model_type == "ensemble":
        filtered_dates, filtered_data = filter_to_trading_days(
            test_dates, [preds, y_raw_test]
        )
        preds, y_raw_test = filtered_data
    else:  # transformer - y_test instead of y_raw_test
        filtered_dates, filtered_data = filter_to_trading_days(
            test_dates, [preds, y_raw_test]
        )
        preds, y_test = filtered_data
        test_dates = filtered_dates
    
    test_dates = filtered_dates
    
    strat_rets = []
    audit_trail = []
    
    for i in range(len(preds)):
        if model_type == "ensemble":
            best_idx = preds[i]
            signal_etf = target_etfs[best_idx].replace('_Ret', '')
            realized_ret = y_raw_test[i][best_idx]
        else:  # transformer
            best_idx = np.argmax(preds[i])
            signal_etf = target_etfs[best_idx].replace('_Ret', '')
            realized_ret = y_test[i][best_idx]
        
        net_ret = realized_ret - (fee_bps / 10000)
        
        strat_rets.append(net_ret)
        
        audit_trail.append({
            'Date': test_dates[i].strftime('%Y-%m-%d'),
            'Signal': signal_etf,
            'Realized': realized_ret,
            'Net_Return': net_ret
        })
    
    strat_rets = np.array(strat_rets)
    
    # Get next trading day signal
    if len(test_dates) > 0:
        last_date = test_dates[-1]
        next_trading_date = get_next_trading_day(last_date)
        
        if len(preds) > 0:
            if model_type == "ensemble":
                next_best_idx = preds[-1]
            else:
                next_best_idx = np.argmax(preds[-1])
            next_signal = target_etfs[next_best_idx].replace('_Ret', '')
        else:
            next_signal = "CASH"
    else:
        from datetime import datetime
        next_trading_date = datetime.now().date()
        next_signal = "CASH"
    
    return strat_rets, audit_trail, next_signal, next_trading_date


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
