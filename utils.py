"""
Utility functions for date handling, NYSE calendar, and time management
"""

from datetime import datetime, timedelta
import pytz

try:
    import pandas_market_calendars as mcal
    NYSE_CALENDAR_AVAILABLE = True
except ImportError:
    NYSE_CALENDAR_AVAILABLE = False


def get_next_trading_day(current_date):
    """Get next valid NYSE trading day (skip weekends and holidays)"""
    if NYSE_CALENDAR_AVAILABLE:
        try:
            nyse = mcal.get_calendar('NYSE')
            schedule = nyse.schedule(
                start_date=current_date,
                end_date=current_date + timedelta(days=10)
            )
            if len(schedule) > 0:
                next_day = schedule.index[0].date()
                if next_day == current_date.date():
                    if len(schedule) > 1:
                        return schedule.index[1].date()
                return next_day
        except Exception as e:
            pass
    
    # Fallback: simple weekend skip
    next_day = current_date + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day.date()


def get_est_time():
    """Get current time in US Eastern timezone"""
    return datetime.now(pytz.timezone('US/Eastern'))


def is_sync_window():
    """Check if current time is within sync windows (7-8am or 7-8pm EST)"""
    now_est = get_est_time()
    return (7 <= now_est.hour < 8) or (19 <= now_est.hour < 20)


def filter_to_trading_days(dates, data_arrays):
    """
    Filter dates and corresponding data arrays to only NYSE trading days
    
    Args:
        dates: pandas DatetimeIndex
        data_arrays: list of numpy arrays to filter (same length as dates)
    
    Returns:
        filtered_dates, filtered_arrays (list)
    """
    if not NYSE_CALENDAR_AVAILABLE:
        return dates, data_arrays
    
    try:
        import pandas as pd
        import numpy as np
        
        nyse = mcal.get_calendar('NYSE')
        trading_schedule = nyse.schedule(
            start_date=dates[0].strftime('%Y-%m-%d'),
            end_date=dates[-1].strftime('%Y-%m-%d')
        )
        valid_trading_days = trading_schedule.index.normalize()
        
        if valid_trading_days.tz is not None:
            valid_trading_days = valid_trading_days.tz_localize(None)
        
        trading_day_mask = dates.isin(valid_trading_days)
        filtered_dates = dates[trading_day_mask]
        
        # Convert mask properly
        if isinstance(trading_day_mask, pd.Series):
            mask_array = trading_day_mask.values
        elif hasattr(trading_day_mask, 'to_numpy'):
            mask_array = trading_day_mask.to_numpy()
        else:
            mask_array = np.array(trading_day_mask)
        
        # Apply mask to all data arrays
        filtered_arrays = [arr[mask_array] for arr in data_arrays]
        
        return filtered_dates, filtered_arrays
        
    except Exception as e:
        print(f"Warning: NYSE calendar filter failed: {e}")
        return dates, data_arrays
