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
    """
    Get the next valid NYSE trading day for signal display.
    
    Logic:
      - If today IS a trading day AND the market has NOT yet opened (before 9:30am EST),
        return TODAY (the signal is for today's session).
      - Otherwise return the NEXT trading day after today.
    
    NOTE: current_date (last date in test set) is used only as a lower bound;
          we always anchor to today's actual date.
    """
    now_est = get_est_time()
    today = now_est.date()

    market_open_hour = 9
    market_open_minute = 30

    # Market has not opened yet if before 9:30 AM EST
    market_not_yet_open = (
        now_est.hour < market_open_hour or
        (now_est.hour == market_open_hour and now_est.minute < market_open_minute)
    )

    if NYSE_CALENDAR_AVAILABLE:
        try:
            nyse = mcal.get_calendar('NYSE')
            # Check a window starting from today
            schedule = nyse.schedule(
                start_date=today,
                end_date=today + timedelta(days=10)
            )
            if len(schedule) > 0:
                first_trading_day = schedule.index[0].date()

                # If today is a trading day and market hasn't opened → return today
                if first_trading_day == today and market_not_yet_open:
                    return today

                # Otherwise return the next trading day after today
                for ts in schedule.index:
                    d = ts.date()
                    if d > today:
                        return d

                # Fallback: last date in schedule
                return schedule.index[-1].date()
        except Exception as e:
            print(f"NYSE calendar error: {e}")

    # Fallback: simple weekend skip
    candidate = today if market_not_yet_open else today + timedelta(days=1)
    while candidate.weekday() >= 5:  # 5=Sat, 6=Sun
        candidate += timedelta(days=1)
    return candidate


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
