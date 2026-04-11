import pandas as pd
import holidays


def get_non_workdays(start_date: str, end_date: str) -> set:
    """
    Return a set of dates that are not normal workdays in Sweden.

    Includes weekends, public holidays, and common bridge days
    (e.g. Friday after Ascension Thursday) where industrial demand
    drops significantly.

    Args:
        start_date: Start date string (YYYY-MM-DD or date object).
        end_date:   End date string (YYYY-MM-DD or date object).

    Returns:
        Set of date objects that are non-workdays.
    """
    dates = pd.date_range(start_date, end_date, freq="D")
    years = sorted(dates.year.unique())
    se_holidays = holidays.Sweden(years=years)

    non_workdays = set()

    for d in dates:
        # Weekends
        if d.dayofweek >= 5:
            non_workdays.add(d.date())
        # Public holidays
        elif d in se_holidays:
            non_workdays.add(d.date())

    # Bridge days: Friday after Ascension Thursday
    for year in years:
        for dt, name in sorted(se_holidays.items()):
            if "Kristi" in name and dt.year == year:
                bridge = dt + pd.Timedelta(days=1)
                non_workdays.add(bridge.date() if hasattr(bridge, 'date') else bridge)

    return non_workdays
