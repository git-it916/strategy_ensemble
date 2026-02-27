"""
Lifecycle Manager

Manages alpha lifecycle transitions:
  pending → paper → live → killed

Checks promotion conditions (paper period elapsed)
and kill conditions (consecutive losses, low Sharpe).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from src.openclaw.config import EXECUTION_POLICY, ExecutionPolicy
from src.openclaw.registry.alpha_registry import AlphaEntry, AlphaRegistry
from src.openclaw.registry.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


class LifecycleManager:
    """
    Manages alpha lifecycle transitions with automatic
    promotion and kill switch enforcement.
    """

    def __init__(
        self,
        registry: AlphaRegistry,
        performance_tracker: PerformanceTracker,
        policy: ExecutionPolicy | None = None,
        notifier=None,
    ):
        self.registry = registry
        self.tracker = performance_tracker
        self.policy = policy or EXECUTION_POLICY
        self.notifier = notifier  # TelegramNotifier (optional)

    def check_promotions(self) -> list[str]:
        """
        Check paper-trading alphas eligible for live promotion.

        Conditions:
          - Paper trading for >= paper_trade_days
          - Paper-period Sharpe >= quality gate min_sharpe (relaxed to 0.5)
          - Active alpha count < max_active_alphas

        Returns:
            List of promoted alpha names.
        """
        promoted = []
        paper_alphas = self.registry.get_paper()

        for entry in paper_alphas:
            if not entry.paper_start_date:
                continue

            start = datetime.fromisoformat(entry.paper_start_date)
            days_elapsed = (datetime.now() - start).days

            if days_elapsed < self.policy.paper_trade_days:
                continue

            # Check paper-period performance
            sharpe = self.tracker.get_rolling_sharpe(
                entry.name, window=days_elapsed
            )

            if sharpe < 0.5:  # relaxed threshold for paper period
                logger.info(
                    f"Paper alpha {entry.name} has low Sharpe ({sharpe:.2f}), "
                    f"keeping in paper"
                )
                continue

            # Check capacity
            if self.registry.active_count >= self.policy.max_active_alphas:
                logger.info(
                    f"Cannot promote {entry.name}: "
                    f"at max active alphas ({self.policy.max_active_alphas})"
                )
                continue

            self.registry.activate(entry.name, status="live")
            promoted.append(entry.name)

            msg = (
                f"Alpha promoted to LIVE: {entry.name}\n"
                f"Paper days: {days_elapsed}\n"
                f"Paper Sharpe: {sharpe:.2f}"
            )
            logger.info(msg)
            self._notify(msg)

        return promoted

    def check_kill_conditions(self) -> list[str]:
        """
        Check all live alphas for kill conditions.

        Kill triggers:
          1. Consecutive loss days >= kill_consecutive_loss_days (5)
          2. Rolling Sharpe < kill_min_sharpe (0.2)

        Returns:
            List of killed alpha names.
        """
        killed = []
        live_alphas = self.registry.get_active()

        for entry in live_alphas:
            kill_reason = self._evaluate_kill(entry)

            if kill_reason:
                self.registry.kill(entry.name, reason=kill_reason)
                killed.append(entry.name)

                msg = (
                    f"Alpha KILLED: {entry.name}\n"
                    f"Reason: {kill_reason}"
                )
                logger.warning(msg)
                self._notify(msg)

        return killed

    def _evaluate_kill(self, entry: AlphaEntry) -> str | None:
        """
        Evaluate kill conditions for a single alpha.

        Returns:
            Kill reason string, or None if alpha is healthy.
        """
        # Check consecutive loss days
        consec_loss = self.tracker.get_consecutive_loss_days(entry.name)
        if consec_loss >= self.policy.kill_consecutive_loss_days:
            return (
                f"Consecutive loss days: {consec_loss} "
                f"(threshold: {self.policy.kill_consecutive_loss_days})"
            )

        # Check rolling Sharpe
        sharpe = self.tracker.get_rolling_sharpe(entry.name, window=63)
        returns = self.tracker.get_daily_returns(entry.name)

        # Only check Sharpe if we have enough data (at least 21 days)
        if len(returns) >= 21 and sharpe < self.policy.kill_min_sharpe:
            return (
                f"Rolling Sharpe: {sharpe:.2f} "
                f"(threshold: {self.policy.kill_min_sharpe})"
            )

        return None

    def daily_lifecycle_check(self) -> dict:
        """
        Run all lifecycle checks. Should be called once per day.

        Returns:
            Summary of actions taken.
        """
        logger.info("Running daily lifecycle check...")

        promoted = self.check_promotions()
        killed = self.check_kill_conditions()

        # Also check paper alphas for early kill
        paper_killed = []
        for entry in self.registry.get_paper():
            kill_reason = self._evaluate_kill(entry)
            if kill_reason:
                self.registry.kill(entry.name, reason=f"[paper] {kill_reason}")
                paper_killed.append(entry.name)
                msg = f"Paper alpha KILLED: {entry.name}\nReason: {kill_reason}"
                logger.warning(msg)
                self._notify(msg)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "promoted": promoted,
            "killed": killed,
            "paper_killed": paper_killed,
            "active_count": self.registry.active_count,
            "paper_count": len(self.registry.get_paper()),
            "total_count": self.registry.total_count,
        }

        logger.info(
            f"Lifecycle check complete: "
            f"promoted={len(promoted)}, killed={len(killed) + len(paper_killed)}, "
            f"active={summary['active_count']}"
        )

        return summary

    def get_health_report(self) -> list[dict]:
        """Get health status of all active + paper alphas."""
        report = []

        for entry in self.registry.get_active() + self.registry.get_paper():
            summary = self.tracker.get_summary(entry.name)
            kill_reason = self._evaluate_kill(entry)

            report.append({
                "name": entry.name,
                "status": entry.status,
                "sharpe": summary["sharpe"],
                "mdd": summary["mdd"],
                "win_rate": summary["win_rate"],
                "consecutive_loss_days": summary["consecutive_loss_days"],
                "n_days": summary["n_days"],
                "total_return": summary["total_return"],
                "healthy": kill_reason is None,
                "warning": kill_reason,
            })

        return report

    def _notify(self, message: str) -> None:
        """Send notification if notifier is available."""
        if self.notifier:
            try:
                self.notifier.send_message(message)
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")
