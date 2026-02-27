"""
Telegram Message Formatters

HTML-formatted message builders for OpenClaw Telegram notifications.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


class TelegramFormatter:
    """Build formatted messages for Telegram (HTML parse mode)."""

    @staticmethod
    def research_started(query: str) -> str:
        return (
            f"<b>Research Session Started</b>\n\n"
            f"Query: {query}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
            f"Searching for alpha ideas..."
        )

    @staticmethod
    def research_progress(step: str, detail: str = "") -> str:
        return f"<b>[Research]</b> {step}\n{detail}" if detail else f"<b>[Research]</b> {step}"

    @staticmethod
    def alpha_approval_request(
        alpha_name: str,
        hypothesis: str,
        source: str,
        oos_sharpe: float,
        oos_mdd: float,
        oos_return: float,
        is_sharpe: float,
        turnover: float,
        correlations: dict[str, float],
        gate_passed: bool,
    ) -> tuple[str, dict]:
        """
        Build approval request message with inline keyboard.

        Returns:
            (message_text, reply_markup_dict)
        """
        status_icon = "PASSED" if gate_passed else "FAILED"

        lines = [
            f"<b>New Alpha: {alpha_name}</b>",
            f"Status: {status_icon}",
            "",
            f"<b>Hypothesis:</b> {hypothesis[:150]}",
            f"Source: {source[:80]}",
            "",
            "<b>Performance (OOS):</b>",
            f"  Sharpe: {oos_sharpe:.2f}",
            f"  Return: {oos_return:.2%}",
            f"  MDD: {oos_mdd:.2%}",
            f"  Turnover: {turnover:.2%}/day",
            f"  IS Sharpe: {is_sharpe:.2f}",
            "",
        ]

        if correlations:
            lines.append("<b>Correlations:</b>")
            for name, corr in sorted(
                correlations.items(), key=lambda x: abs(x[1]), reverse=True
            )[:5]:
                lines.append(f"  {name}: {corr:+.3f}")
            lines.append("")

        if gate_passed:
            lines.append("Approve for paper trading?")
        else:
            lines.append("Gates FAILED. Force approve?")

        text = "\n".join(lines)

        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "Approve", "callback_data": f"approve_{alpha_name}"},
                    {"text": "Reject", "callback_data": f"reject_{alpha_name}"},
                ]
            ]
        }

        return text, keyboard

    @staticmethod
    def alpha_killed(
        alpha_name: str,
        reason: str,
        total_return: float,
        sharpe: float,
        n_days: int,
    ) -> str:
        return (
            f"<b>Alpha KILLED: {alpha_name}</b>\n\n"
            f"Reason: {reason}\n"
            f"Days active: {n_days}\n"
            f"Total return: {total_return:.2%}\n"
            f"Sharpe: {sharpe:.2f}"
        )

    @staticmethod
    def alpha_promoted(alpha_name: str, paper_days: int, paper_sharpe: float) -> str:
        return (
            f"<b>Alpha PROMOTED to LIVE: {alpha_name}</b>\n\n"
            f"Paper days: {paper_days}\n"
            f"Paper Sharpe: {paper_sharpe:.2f}"
        )

    @staticmethod
    def status_report(
        active_alphas: list[dict],
        paper_alphas: list[dict],
        total_pnl: float,
    ) -> str:
        lines = [
            "<b>== OpenClaw Status ==</b>",
            f"Total PnL: {total_pnl:.2%}",
            "",
        ]

        if active_alphas:
            lines.append(f"<b>Live ({len(active_alphas)}):</b>")
            for a in active_alphas:
                lines.append(
                    f"  {a['name']} | "
                    f"Sh {a.get('sharpe', 0):.2f} | "
                    f"MDD {a.get('mdd', 0):.2%} | "
                    f"Lev {a.get('leverage', 1):.1f}x | "
                    f"Wt {a.get('weight', 0):.1%}"
                )
            lines.append("")

        if paper_alphas:
            lines.append(f"<b>Paper ({len(paper_alphas)}):</b>")
            for a in paper_alphas:
                lines.append(
                    f"  {a['name']} | Sh {a.get('sharpe', 0):.2f} | "
                    f"D{a.get('n_days', 0)}"
                )

        if not active_alphas and not paper_alphas:
            lines.append("No active or paper alphas.")

        return "\n".join(lines)

    @staticmethod
    def daily_summary(
        date: datetime,
        active_count: int,
        total_pnl_today: float,
        total_pnl_cumulative: float,
        alpha_details: list[dict],
    ) -> str:
        pnl_icon = "+" if total_pnl_today >= 0 else ""
        lines = [
            f"<b>Daily Report — {date.strftime('%Y-%m-%d')}</b>",
            "",
            f"Active alphas: {active_count}",
            f"Today PnL: {pnl_icon}{total_pnl_today:.2%}",
            f"Cumulative: {total_pnl_cumulative:.2%}",
            "",
        ]

        if alpha_details:
            lines.append("<b>Per-Alpha:</b>")
            for a in alpha_details:
                icon = "+" if a.get("pnl", 0) >= 0 else ""
                lines.append(
                    f"  {a['name']}: {icon}{a.get('pnl', 0):.2%}"
                )

        return "\n".join(lines)

    @staticmethod
    def mutation_result(
        alpha_name: str,
        mutation_type: str,
        old_sharpe: float,
        new_sharpe: float,
    ) -> tuple[str, dict]:
        improvement = new_sharpe - old_sharpe
        text = (
            f"<b>Mutation: {alpha_name}</b>\n\n"
            f"Type: {mutation_type}\n"
            f"Old Sharpe: {old_sharpe:.2f}\n"
            f"New Sharpe: {new_sharpe:.2f} ({improvement:+.2f})\n\n"
            f"Approve variant for paper trading?"
        )

        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "Approve", "callback_data": f"approve_{alpha_name}_v2"},
                    {"text": "Reject", "callback_data": f"reject_{alpha_name}_v2"},
                ]
            ]
        }

        return text, keyboard

    @staticmethod
    def error_alert(error: str, context: str = "") -> str:
        return (
            f"<b>OpenClaw Error</b>\n\n"
            f"Context: {context}\n"
            f"Error: {error}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
