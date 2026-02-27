"""
Summary Builder

Generate executive summaries for Telegram approval.
"""

from __future__ import annotations

from typing import Any


class SummaryBuilder:
    """Build formatted HTML summaries for Telegram approval messages."""

    def build_approval_summary(
        self,
        alpha_name: str,
        hypothesis: str,
        source_url: str,
        source_title: str,
        is_metrics: dict[str, float],
        oos_metrics: dict[str, float],
        correlations: dict[str, float],
        gate_failures: list[str],
        gate_warnings: list[str],
        turnover: float,
        code_preview: str = "",
    ) -> str:
        """
        Build approval summary for Telegram.

        Returns:
            HTML-formatted string for Telegram send_message(parse_mode="HTML").
        """
        # Header
        lines = [
            f"<b>New Alpha: {alpha_name}</b>",
            "",
        ]

        # Source
        if source_title:
            lines.append(f"Source: {source_title}")
        if source_url:
            lines.append(f"URL: {source_url}")
        lines.append("")

        # Hypothesis
        if hypothesis:
            lines.append(f"<b>Hypothesis:</b> {hypothesis[:200]}")
            lines.append("")

        # Performance metrics
        lines.append("<b>== Performance ==</b>")
        lines.append(
            f"IS  Sharpe: {is_metrics.get('sharpe_ratio', 0):.2f} | "
            f"Return: {is_metrics.get('total_return', 0):.2%} | "
            f"MDD: {is_metrics.get('max_drawdown', 0):.2%}"
        )
        lines.append(
            f"OOS Sharpe: {oos_metrics.get('sharpe_ratio', 0):.2f} | "
            f"Return: {oos_metrics.get('total_return', 0):.2%} | "
            f"MDD: {oos_metrics.get('max_drawdown', 0):.2%}"
        )
        lines.append(f"Turnover: {turnover:.2%}/day")
        lines.append(
            f"Win Rate: {oos_metrics.get('win_rate', 0):.1%} | "
            f"Profit Factor: {oos_metrics.get('profit_factor', 0):.2f}"
        )
        lines.append("")

        # Correlations
        if correlations:
            lines.append("<b>== Correlations ==</b>")
            for name, corr in sorted(
                correlations.items(), key=lambda x: abs(x[1]), reverse=True
            ):
                flag = " (!)" if abs(corr) > 0.25 else ""
                lines.append(f"  {name}: {corr:+.3f}{flag}")
            lines.append("")

        # Gate results
        if gate_failures:
            lines.append("<b>== FAILURES ==</b>")
            for f in gate_failures:
                lines.append(f"  {f}")
            lines.append("")

        if gate_warnings:
            lines.append("<b>== Warnings ==</b>")
            for w in gate_warnings:
                lines.append(f"  {w}")
            lines.append("")

        # Status
        passed = len(gate_failures) == 0
        status = "PASSED - Approve for paper trading?" if passed else "FAILED"
        lines.append(f"<b>Status: {status}</b>")

        return "\n".join(lines)

    def build_kill_summary(
        self,
        alpha_name: str,
        reason: str,
        performance: dict[str, Any],
    ) -> str:
        """Build kill notification summary."""
        lines = [
            f"<b>Alpha KILLED: {alpha_name}</b>",
            "",
            f"<b>Reason:</b> {reason}",
            "",
            f"Days active: {performance.get('n_days', 0)}",
            f"Total return: {performance.get('total_return', 0):.2%}",
            f"Sharpe: {performance.get('sharpe', 0):.2f}",
            f"MDD: {performance.get('mdd', 0):.2%}",
            f"Consecutive loss days: {performance.get('consecutive_loss_days', 0)}",
        ]

        return "\n".join(lines)

    def build_status_summary(
        self,
        active_alphas: list[dict],
        paper_alphas: list[dict],
    ) -> str:
        """Build status overview for /status command."""
        lines = ["<b>== OpenClaw Status ==</b>", ""]

        if active_alphas:
            lines.append(f"<b>Live ({len(active_alphas)}):</b>")
            for a in active_alphas:
                lines.append(
                    f"  {a['name']} | "
                    f"Sharpe {a.get('sharpe', 0):.2f} | "
                    f"MDD {a.get('mdd', 0):.2%} | "
                    f"Lev {a.get('leverage', 1):.1f}x | "
                    f"Wt {a.get('weight', 0):.1%}"
                )
            lines.append("")

        if paper_alphas:
            lines.append(f"<b>Paper ({len(paper_alphas)}):</b>")
            for a in paper_alphas:
                lines.append(
                    f"  {a['name']} | "
                    f"Sharpe {a.get('sharpe', 0):.2f} | "
                    f"Days {a.get('n_days', 0)}"
                )
            lines.append("")

        if not active_alphas and not paper_alphas:
            lines.append("No active or paper alphas.")

        return "\n".join(lines)

    def build_mutation_summary(
        self,
        alpha_name: str,
        mutation_type: str,
        original_sharpe: float,
        new_sharpe: float,
        details: dict[str, Any],
    ) -> str:
        """Build mutation result summary for Telegram."""
        improvement = new_sharpe - original_sharpe
        lines = [
            f"<b>Alpha Mutation: {alpha_name}</b>",
            "",
            f"Type: {mutation_type}",
            f"Original Sharpe: {original_sharpe:.2f}",
            f"New Sharpe: {new_sharpe:.2f} ({improvement:+.2f})",
            "",
        ]

        if mutation_type == "parameter_sweep":
            lines.append(f"Original params: {details.get('original_params', {})}")
            lines.append(f"New params: {details.get('best_params', {})}")
        elif mutation_type == "feature_mutation":
            lines.append(f"Operation: {details.get('operation', 'unknown')}")
            lines.append(f"Features changed: {details.get('features_changed', [])}")

        lines.append("")
        lines.append("Approve for paper trading?")

        return "\n".join(lines)
