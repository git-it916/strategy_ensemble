"""
Backtest Report Generation

Create comprehensive backtest reports and visualizations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common import get_logger
from .engine import BacktestResult
from .metrics import (
    calculate_drawdown_analysis,
    calculate_monthly_returns,
    calculate_rolling_metrics,
)

logger = get_logger(__name__)


class BacktestReport:
    """
    Generate comprehensive backtest reports.

    Outputs:
        - Text summary
        - CSV exports
        - HTML report (optional)
    """

    def __init__(self, result: BacktestResult):
        """
        Initialize report generator.

        Args:
            result: BacktestResult from backtest engine
        """
        self.result = result

    def generate_summary(self) -> str:
        """Generate text summary of backtest results."""
        m = self.result.metrics

        lines = [
            "=" * 60,
            "BACKTEST SUMMARY",
            "=" * 60,
            "",
            "Configuration:",
            f"  Period: {self.result.config.start_date.date()} to {self.result.config.end_date.date()}",
            f"  Initial Capital: {self.result.config.initial_capital:,.0f}",
            f"  Rebalance Frequency: {self.result.config.rebalance_frequency}",
            f"  Transaction Cost: {self.result.config.transaction_cost_bps:.1f} bps",
            "",
            "Performance:",
            f"  Total Return: {m.get('total_return', 0):.2%}",
            f"  Annual Return: {m.get('annual_return', 0):.2%}",
            f"  Annual Volatility: {m.get('annual_volatility', 0):.2%}",
            f"  Sharpe Ratio: {m.get('sharpe_ratio', 0):.2f}",
            f"  Sortino Ratio: {m.get('sortino_ratio', 0):.2f}",
            f"  Calmar Ratio: {m.get('calmar_ratio', 0):.2f}",
            "",
            "Risk:",
            f"  Max Drawdown: {m.get('max_drawdown', 0):.2%}",
            f"  Avg Drawdown: {m.get('avg_drawdown', 0):.2%}",
            f"  Max DD Duration: {m.get('max_drawdown_duration', 0):.0f} days",
            f"  VaR (95%): {m.get('var_95', 0):.2%}",
            f"  CVaR (95%): {m.get('cvar_95', 0):.2%}",
            "",
            "Trading:",
            f"  Win Rate: {m.get('win_rate', 0):.2%}",
            f"  Profit Factor: {m.get('profit_factor', 0):.2f}",
            f"  Avg Win: {m.get('avg_win', 0):.4f}",
            f"  Avg Loss: {m.get('avg_loss', 0):.4f}",
            f"  Win/Loss Ratio: {m.get('win_loss_ratio', 0):.2f}",
            "",
            "Statistics:",
            f"  Skewness: {m.get('skewness', 0):.2f}",
            f"  Kurtosis: {m.get('kurtosis', 0):.2f}",
            f"  Trading Days: {m.get('n_days', 0)}",
            "",
            "Final:",
            f"  Final Value: {m.get('final_value', 0):,.0f}",
            f"  P&L: {m.get('profit_loss', 0):,.0f}",
            "",
            "=" * 60,
        ]

        # Add benchmark comparison if available
        if "alpha" in m:
            lines.extend([
                "",
                "Benchmark Comparison:",
                f"  Alpha: {m.get('alpha', 0):.2%}",
                f"  Beta: {m.get('beta', 0):.2f}",
                f"  Information Ratio: {m.get('information_ratio', 0):.2f}",
                f"  Tracking Error: {m.get('tracking_error', 0):.2%}",
                f"  Correlation: {m.get('benchmark_correlation', 0):.2f}",
                "",
            ])

        return "\n".join(lines)

    def generate_monthly_table(self) -> pd.DataFrame:
        """Generate monthly returns table."""
        return calculate_monthly_returns(self.result.returns_series)

    def generate_drawdown_table(self) -> pd.DataFrame:
        """Generate drawdown analysis table."""
        return calculate_drawdown_analysis(self.result.returns_series)

    def generate_rolling_metrics(self, window: int = 252) -> pd.DataFrame:
        """Generate rolling metrics."""
        return calculate_rolling_metrics(self.result.returns_series, window)

    def export_to_csv(self, output_dir: str | Path) -> dict[str, Path]:
        """
        Export all results to CSV files.

        Args:
            output_dir: Output directory

        Returns:
            Dictionary of exported file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exports = {}

        # Returns
        returns_path = output_dir / "returns.csv"
        self.result.returns_series.to_csv(returns_path)
        exports["returns"] = returns_path

        # Trades
        if not self.result.trades_df.empty:
            trades_path = output_dir / "trades.csv"
            self.result.trades_df.to_csv(trades_path, index=False)
            exports["trades"] = trades_path

        # Weights
        if not self.result.weights_df.empty:
            weights_path = output_dir / "weights.csv"
            self.result.weights_df.to_csv(weights_path, index=False)
            exports["weights"] = weights_path

        # Metrics
        metrics_path = output_dir / "metrics.csv"
        pd.Series(self.result.metrics).to_csv(metrics_path)
        exports["metrics"] = metrics_path

        # Monthly returns
        monthly = self.generate_monthly_table()
        if not monthly.empty:
            monthly_path = output_dir / "monthly_returns.csv"
            monthly.to_csv(monthly_path)
            exports["monthly_returns"] = monthly_path

        # Drawdowns
        drawdowns = self.generate_drawdown_table()
        if not drawdowns.empty:
            drawdowns_path = output_dir / "drawdowns.csv"
            drawdowns.to_csv(drawdowns_path, index=False)
            exports["drawdowns"] = drawdowns_path

        # Summary
        summary_path = output_dir / "summary.txt"
        with open(summary_path, "w") as f:
            f.write(self.generate_summary())
        exports["summary"] = summary_path

        logger.info(f"Exported {len(exports)} files to {output_dir}")

        return exports

    def generate_html_report(self, output_path: str | Path | None = None) -> str:
        """
        Generate HTML report.

        Args:
            output_path: Optional path to save HTML file

        Returns:
            HTML string
        """
        m = self.result.metrics
        monthly_df = self.generate_monthly_table()
        dd_df = self.generate_drawdown_table()

        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "  <title>Backtest Report</title>",
            "  <style>",
            "    body { font-family: Arial, sans-serif; margin: 20px; }",
            "    h1, h2 { color: #333; }",
            "    table { border-collapse: collapse; margin: 10px 0; }",
            "    th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }",
            "    th { background-color: #4CAF50; color: white; }",
            "    .metric-card { display: inline-block; margin: 10px; padding: 15px; ",
            "                   border: 1px solid #ddd; border-radius: 5px; }",
            "    .metric-value { font-size: 24px; font-weight: bold; }",
            "    .metric-label { color: #666; }",
            "    .positive { color: green; }",
            "    .negative { color: red; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>Backtest Report</h1>",
            f"  <p>Period: {self.result.config.start_date.date()} to {self.result.config.end_date.date()}</p>",
            "",
            "  <h2>Key Metrics</h2>",
            "  <div class='metric-cards'>",
        ]

        # Metric cards
        key_metrics = [
            ("Total Return", f"{m.get('total_return', 0):.2%}", m.get("total_return", 0) >= 0),
            ("Sharpe Ratio", f"{m.get('sharpe_ratio', 0):.2f}", m.get("sharpe_ratio", 0) >= 1),
            ("Max Drawdown", f"{m.get('max_drawdown', 0):.2%}", True),
            ("Win Rate", f"{m.get('win_rate', 0):.2%}", m.get("win_rate", 0) >= 0.5),
        ]

        for label, value, is_positive in key_metrics:
            css_class = "positive" if is_positive else "negative"
            html_parts.append(
                f"    <div class='metric-card'>"
                f"<div class='metric-value {css_class}'>{value}</div>"
                f"<div class='metric-label'>{label}</div></div>"
            )

        html_parts.append("  </div>")

        # All metrics table
        html_parts.extend([
            "",
            "  <h2>Detailed Metrics</h2>",
            "  <table>",
            "    <tr><th>Metric</th><th>Value</th></tr>",
        ])

        for key, value in m.items():
            if isinstance(value, float):
                if "return" in key or "drawdown" in key or "rate" in key:
                    formatted = f"{value:.2%}"
                else:
                    formatted = f"{value:.4f}"
            else:
                formatted = str(value)
            html_parts.append(f"    <tr><td>{key}</td><td>{formatted}</td></tr>")

        html_parts.append("  </table>")

        # Monthly returns
        if not monthly_df.empty:
            html_parts.extend([
                "",
                "  <h2>Monthly Returns</h2>",
                monthly_df.to_html(classes="monthly-table", float_format=lambda x: f"{x:.2%}" if pd.notna(x) else ""),
            ])

        # Drawdowns
        if not dd_df.empty:
            html_parts.extend([
                "",
                "  <h2>Worst Drawdowns</h2>",
                dd_df.to_html(classes="drawdown-table", index=False),
            ])

        html_parts.extend([
            "",
            f"  <p><small>Generated: {self.result.run_timestamp}</small></p>",
            "</body>",
            "</html>",
        ])

        html = "\n".join(html_parts)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(html)
            logger.info(f"Saved HTML report to {output_path}")

        return html


def compare_strategies(
    results: dict[str, BacktestResult],
    benchmark_name: str | None = None,
) -> pd.DataFrame:
    """
    Compare multiple strategy results.

    Args:
        results: Dictionary of strategy_name -> BacktestResult
        benchmark_name: Name of benchmark strategy

    Returns:
        Comparison DataFrame
    """
    rows = []

    benchmark_returns = None
    if benchmark_name and benchmark_name in results:
        benchmark_returns = results[benchmark_name].returns_series

    for name, result in results.items():
        m = result.metrics

        row = {
            "Strategy": name,
            "Total Return": m.get("total_return", 0),
            "Annual Return": m.get("annual_return", 0),
            "Volatility": m.get("annual_volatility", 0),
            "Sharpe": m.get("sharpe_ratio", 0),
            "Sortino": m.get("sortino_ratio", 0),
            "Max DD": m.get("max_drawdown", 0),
            "Win Rate": m.get("win_rate", 0),
            "Profit Factor": m.get("profit_factor", 0),
        }

        # Add benchmark comparison
        if benchmark_returns is not None and name != benchmark_name:
            aligned = pd.concat(
                [result.returns_series, benchmark_returns], axis=1
            ).dropna()

            if len(aligned) > 20:
                excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
                row["Excess Return"] = excess.sum()
                row["Tracking Error"] = excess.std() * np.sqrt(252)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by Sharpe
    df = df.sort_values("Sharpe", ascending=False).reset_index(drop=True)

    return df


def print_report(result: BacktestResult) -> None:
    """Print backtest report to console."""
    report = BacktestReport(result)
    print(report.generate_summary())


def export_report(
    result: BacktestResult,
    output_dir: str | Path,
    include_html: bool = True,
) -> dict[str, Path]:
    """
    Export complete backtest report.

    Args:
        result: BacktestResult
        output_dir: Output directory
        include_html: Whether to generate HTML report

    Returns:
        Dictionary of exported paths
    """
    report = BacktestReport(result)
    exports = report.export_to_csv(output_dir)

    if include_html:
        html_path = Path(output_dir) / "report.html"
        report.generate_html_report(html_path)
        exports["html"] = html_path

    return exports
