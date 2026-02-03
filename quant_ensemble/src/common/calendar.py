"""
Trading Calendar Utilities

Provides trading calendar functionality for Korean markets.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Sequence

import numpy as np
import pandas as pd


class TradingCalendar:
    """
    Trading calendar for Korean markets (KRX).

    Handles trading days, holidays, and market hours.
    """

    # Korean market holidays (sample - should be updated annually)
    KOREAN_HOLIDAYS_2024 = [
        "2024-01-01",  # New Year
        "2024-02-09",  # Lunar New Year
        "2024-02-10",  # Lunar New Year
        "2024-02-11",  # Lunar New Year
        "2024-02-12",  # Lunar New Year (substitute)
        "2024-03-01",  # Independence Day
        "2024-04-10",  # Election Day
        "2024-05-05",  # Children's Day
        "2024-05-06",  # Children's Day (substitute)
        "2024-05-15",  # Buddha's Birthday
        "2024-06-06",  # Memorial Day
        "2024-08-15",  # Liberation Day
        "2024-09-16",  # Chuseok
        "2024-09-17",  # Chuseok
        "2024-09-18",  # Chuseok
        "2024-10-03",  # National Foundation Day
        "2024-10-09",  # Hangul Day
        "2024-12-25",  # Christmas
    ]

    def __init__(
        self,
        start_date: str | pd.Timestamp = "2010-01-01",
        end_date: str | pd.Timestamp = "2030-12-31",
        holidays: list[str] | None = None,
    ):
        """
        Initialize trading calendar.

        Args:
            start_date: Calendar start date
            end_date: Calendar end date
            holidays: List of holiday dates in YYYY-MM-DD format
        """
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)

        # Combine provided holidays with defaults
        all_holidays = set(self.KOREAN_HOLIDAYS_2024)
        if holidays:
            all_holidays.update(holidays)

        self.holidays = {pd.Timestamp(h) for h in all_holidays}

        # Generate trading days
        self._trading_days = self._generate_trading_days()
        self._trading_days_set = set(self._trading_days)

    def _generate_trading_days(self) -> pd.DatetimeIndex:
        """Generate all trading days between start and end date."""
        # Get all business days
        all_days = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq="B",  # Business days (Mon-Fri)
        )

        # Remove holidays
        trading_days = all_days[~all_days.isin(self.holidays)]

        return trading_days

    def is_trading_day(self, date: str | pd.Timestamp) -> bool:
        """
        Check if a date is a trading day.

        Args:
            date: Date to check

        Returns:
            True if trading day, False otherwise
        """
        date = pd.Timestamp(date).normalize()
        return date in self._trading_days_set

    def get_trading_days(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DatetimeIndex:
        """
        Get trading days in a date range.

        Args:
            start: Start date (default: calendar start)
            end: End date (default: calendar end)

        Returns:
            DatetimeIndex of trading days
        """
        start = pd.Timestamp(start) if start else self.start_date
        end = pd.Timestamp(end) if end else self.end_date

        mask = (self._trading_days >= start) & (self._trading_days <= end)
        return self._trading_days[mask]

    def get_previous_trading_day(
        self,
        date: str | pd.Timestamp,
        n: int = 1,
    ) -> pd.Timestamp:
        """
        Get the nth previous trading day.

        Args:
            date: Reference date
            n: Number of trading days to go back

        Returns:
            Previous trading day
        """
        date = pd.Timestamp(date).normalize()
        trading_days_before = self._trading_days[self._trading_days < date]

        if len(trading_days_before) < n:
            raise ValueError(f"Not enough trading days before {date}")

        return trading_days_before[-n]

    def get_next_trading_day(
        self,
        date: str | pd.Timestamp,
        n: int = 1,
    ) -> pd.Timestamp:
        """
        Get the nth next trading day.

        Args:
            date: Reference date
            n: Number of trading days to go forward

        Returns:
            Next trading day
        """
        date = pd.Timestamp(date).normalize()
        trading_days_after = self._trading_days[self._trading_days > date]

        if len(trading_days_after) < n:
            raise ValueError(f"Not enough trading days after {date}")

        return trading_days_after[n - 1]

    def get_trading_day_offset(
        self,
        date: str | pd.Timestamp,
        offset: int,
    ) -> pd.Timestamp:
        """
        Get trading day with offset from reference date.

        Args:
            date: Reference date
            offset: Number of trading days (positive = forward, negative = backward)

        Returns:
            Trading day at offset
        """
        if offset > 0:
            return self.get_next_trading_day(date, offset)
        elif offset < 0:
            return self.get_previous_trading_day(date, -offset)
        else:
            date = pd.Timestamp(date).normalize()
            if self.is_trading_day(date):
                return date
            return self.get_previous_trading_day(date, 1)

    def get_rebalance_dates(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        frequency: str = "weekly",
        day_of_week: str = "monday",
        day_of_month: int = 1,
    ) -> pd.DatetimeIndex:
        """
        Get rebalancing dates based on frequency.

        Args:
            start: Start date
            end: End date
            frequency: 'daily', 'weekly', or 'monthly'
            day_of_week: For weekly, which day (monday, tuesday, etc.)
            day_of_month: For monthly, which day (1-28)

        Returns:
            DatetimeIndex of rebalancing dates
        """
        trading_days = self.get_trading_days(start, end)

        if frequency == "daily":
            return trading_days

        elif frequency == "weekly":
            day_map = {
                "monday": 0,
                "tuesday": 1,
                "wednesday": 2,
                "thursday": 3,
                "friday": 4,
            }
            target_day = day_map.get(day_of_week.lower(), 0)

            rebalance_dates = []
            for date in trading_days:
                if date.dayofweek == target_day:
                    rebalance_dates.append(date)
                elif date.dayofweek > target_day:
                    # Find the next available trading day if target day is holiday
                    week_start = date - timedelta(days=date.dayofweek)
                    target_date = week_start + timedelta(days=target_day)
                    if target_date in self._trading_days_set:
                        if target_date not in rebalance_dates:
                            rebalance_dates.append(target_date)

            return pd.DatetimeIndex(sorted(set(rebalance_dates)))

        elif frequency == "monthly":
            rebalance_dates = []
            current_month = None

            for date in trading_days:
                if current_month != (date.year, date.month):
                    current_month = (date.year, date.month)
                    # Find the first trading day on or after day_of_month
                    target_date = pd.Timestamp(
                        year=date.year,
                        month=date.month,
                        day=min(day_of_month, 28),
                    )
                    while target_date.month == date.month:
                        if target_date in self._trading_days_set:
                            rebalance_dates.append(target_date)
                            break
                        target_date += timedelta(days=1)

            return pd.DatetimeIndex(rebalance_dates)

        else:
            raise ValueError(f"Unknown frequency: {frequency}")

    def count_trading_days(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> int:
        """
        Count trading days between two dates (inclusive).

        Args:
            start: Start date
            end: End date

        Returns:
            Number of trading days
        """
        return len(self.get_trading_days(start, end))


# Global calendar instance
_default_calendar: TradingCalendar | None = None


def get_calendar() -> TradingCalendar:
    """Get the default trading calendar instance."""
    global _default_calendar
    if _default_calendar is None:
        _default_calendar = TradingCalendar()
    return _default_calendar


def set_calendar(calendar: TradingCalendar) -> None:
    """Set the default trading calendar instance."""
    global _default_calendar
    _default_calendar = calendar


def is_trading_day(date: str | pd.Timestamp) -> bool:
    """Check if a date is a trading day using default calendar."""
    return get_calendar().is_trading_day(date)


def get_trading_days(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """Get trading days in a range using default calendar."""
    return get_calendar().get_trading_days(start, end)


def get_previous_trading_day(
    date: str | pd.Timestamp,
    n: int = 1,
) -> pd.Timestamp:
    """Get nth previous trading day using default calendar."""
    return get_calendar().get_previous_trading_day(date, n)


def get_next_trading_day(
    date: str | pd.Timestamp,
    n: int = 1,
) -> pd.Timestamp:
    """Get nth next trading day using default calendar."""
    return get_calendar().get_next_trading_day(date, n)
