"""
Trading Scheduler

Schedule and manage trading jobs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

from ..common import get_logger

logger = get_logger(__name__)


class JobStatus(Enum):
    """Job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ScheduledJob:
    """Scheduled job definition."""

    name: str
    func: Callable[[], Any]
    schedule_time: datetime | None = None  # Specific time
    interval_seconds: int | None = None  # Recurring interval
    run_on_market_open: bool = False
    run_on_market_close: bool = False
    enabled: bool = True
    last_run: datetime | None = None
    last_status: JobStatus = JobStatus.PENDING
    last_result: Any = None
    last_error: str | None = None


@dataclass
class MarketHours:
    """Market trading hours."""

    open_time: str = "09:00"  # KST
    close_time: str = "15:30"  # KST
    pre_open_minutes: int = 30
    post_close_minutes: int = 30
    trading_days: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri


class TradingScheduler:
    """
    Trading job scheduler.

    Features:
        - Time-based scheduling
        - Market hours awareness
        - Interval-based jobs
        - Job dependencies
    """

    def __init__(
        self,
        market_hours: MarketHours | None = None,
    ):
        """
        Initialize scheduler.

        Args:
            market_hours: Market hours configuration
        """
        self.market_hours = market_hours or MarketHours()
        self._jobs: dict[str, ScheduledJob] = {}
        self._running = False

    def add_job(
        self,
        name: str,
        func: Callable[[], Any],
        schedule_time: str | None = None,
        interval_seconds: int | None = None,
        run_on_market_open: bool = False,
        run_on_market_close: bool = False,
    ) -> None:
        """
        Add a scheduled job.

        Args:
            name: Job name
            func: Function to execute
            schedule_time: Specific time (HH:MM format)
            interval_seconds: Run every N seconds
            run_on_market_open: Run at market open
            run_on_market_close: Run at market close
        """
        parsed_time = None
        if schedule_time:
            hour, minute = map(int, schedule_time.split(":"))
            now = datetime.now()
            parsed_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if parsed_time < now:
                parsed_time += timedelta(days=1)

        job = ScheduledJob(
            name=name,
            func=func,
            schedule_time=parsed_time,
            interval_seconds=interval_seconds,
            run_on_market_open=run_on_market_open,
            run_on_market_close=run_on_market_close,
        )

        self._jobs[name] = job
        logger.info(f"Added job: {name}")

    def remove_job(self, name: str) -> bool:
        """Remove a job."""
        if name in self._jobs:
            del self._jobs[name]
            logger.info(f"Removed job: {name}")
            return True
        return False

    def enable_job(self, name: str) -> None:
        """Enable a job."""
        if name in self._jobs:
            self._jobs[name].enabled = True

    def disable_job(self, name: str) -> None:
        """Disable a job."""
        if name in self._jobs:
            self._jobs[name].enabled = False

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now()

        # Check day of week
        if now.weekday() not in self.market_hours.trading_days:
            return False

        # Parse market hours
        open_hour, open_min = map(int, self.market_hours.open_time.split(":"))
        close_hour, close_min = map(int, self.market_hours.close_time.split(":"))

        market_open = now.replace(hour=open_hour, minute=open_min, second=0)
        market_close = now.replace(hour=close_hour, minute=close_min, second=0)

        return market_open <= now <= market_close

    def is_trading_day(self) -> bool:
        """Check if today is a trading day."""
        return datetime.now().weekday() in self.market_hours.trading_days

    def get_market_open_time(self) -> datetime:
        """Get today's market open time."""
        now = datetime.now()
        hour, minute = map(int, self.market_hours.open_time.split(":"))
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    def get_market_close_time(self) -> datetime:
        """Get today's market close time."""
        now = datetime.now()
        hour, minute = map(int, self.market_hours.close_time.split(":"))
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    def _should_run_job(self, job: ScheduledJob) -> bool:
        """Check if job should run now."""
        if not job.enabled:
            return False

        now = datetime.now()

        # Check market open/close triggers
        if job.run_on_market_open:
            market_open = self.get_market_open_time()
            if job.last_run is None or job.last_run.date() < now.date():
                if abs((now - market_open).total_seconds()) < 60:  # Within 1 minute
                    return True

        if job.run_on_market_close:
            market_close = self.get_market_close_time()
            if job.last_run is None or job.last_run.date() < now.date():
                if abs((now - market_close).total_seconds()) < 60:
                    return True

        # Check scheduled time
        if job.schedule_time:
            if job.last_run is None or job.last_run < job.schedule_time:
                if now >= job.schedule_time:
                    return True

        # Check interval
        if job.interval_seconds:
            if job.last_run is None:
                return True
            elapsed = (now - job.last_run).total_seconds()
            if elapsed >= job.interval_seconds:
                return True

        return False

    def _run_job(self, job: ScheduledJob) -> None:
        """Execute a job."""
        logger.info(f"Running job: {job.name}")
        job.last_status = JobStatus.RUNNING

        try:
            result = job.func()
            job.last_result = result
            job.last_status = JobStatus.COMPLETED
            job.last_error = None
            logger.info(f"Job completed: {job.name}")

        except Exception as e:
            job.last_status = JobStatus.FAILED
            job.last_error = str(e)
            logger.error(f"Job failed: {job.name} - {e}")

        job.last_run = datetime.now()

        # Update schedule_time for next day if it was a specific time
        if job.schedule_time and job.schedule_time < datetime.now():
            job.schedule_time += timedelta(days=1)

    def run_once(self) -> list[str]:
        """
        Run one iteration of the scheduler.

        Returns:
            List of job names that were run
        """
        executed = []

        for name, job in self._jobs.items():
            if self._should_run_job(job):
                self._run_job(job)
                executed.append(name)

        return executed

    def start(self, poll_interval: float = 1.0) -> None:
        """
        Start the scheduler loop.

        Args:
            poll_interval: Seconds between checks
        """
        logger.info("Starting scheduler")
        self._running = True

        while self._running:
            try:
                self.run_once()
                time.sleep(poll_interval)

            except KeyboardInterrupt:
                logger.info("Scheduler interrupted")
                break

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(poll_interval)

        logger.info("Scheduler stopped")

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False

    def get_job_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all jobs."""
        status = {}

        for name, job in self._jobs.items():
            status[name] = {
                "enabled": job.enabled,
                "last_run": job.last_run,
                "last_status": job.last_status.value,
                "last_error": job.last_error,
                "schedule_time": job.schedule_time,
                "interval_seconds": job.interval_seconds,
            }

        return status


def create_standard_schedule(
    scheduler: TradingScheduler,
    data_refresh_func: Callable[[], Any],
    signal_generate_func: Callable[[], Any],
    rebalance_func: Callable[[], Any],
    monitoring_func: Callable[[], Any],
) -> None:
    """
    Create standard trading schedule.

    Args:
        scheduler: Scheduler instance
        data_refresh_func: Function to refresh data
        signal_generate_func: Function to generate signals
        rebalance_func: Function to rebalance portfolio
        monitoring_func: Function for monitoring
    """
    # Pre-market data refresh
    scheduler.add_job(
        name="data_refresh",
        func=data_refresh_func,
        schedule_time="08:30",
    )

    # Generate signals at market open
    scheduler.add_job(
        name="signal_generation",
        func=signal_generate_func,
        run_on_market_open=True,
    )

    # Rebalance shortly after open
    scheduler.add_job(
        name="rebalance",
        func=rebalance_func,
        schedule_time="09:10",
    )

    # Monitoring every 5 minutes during market hours
    scheduler.add_job(
        name="monitoring",
        func=monitoring_func,
        interval_seconds=300,
    )

    # End of day report
    scheduler.add_job(
        name="eod_report",
        func=monitoring_func,
        run_on_market_close=True,
    )
