#!/usr/bin/env python
"""
Backtest Script

Run backtests with trained models.

Usage:
    python run_backtest.py --config config/backtest.yaml --model-dir models --data-dir data/processed
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common import get_logger
from backtest import (
    BacktestConfig,
    BacktestEngine,
    print_report,
    export_report,
    run_walk_forward,
)
from portfolio import PortfolioAllocator, PortfolioConstraints, RiskManager, TransactionCostModel

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(model_dir: str) -> object:
    """Load trained ensemble model."""
    model_path = Path(model_dir) / "ensemble.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed data."""
    data_path = Path(data_dir)

    features_df = pd.read_parquet(data_path / "features.parquet")
    labels_df = pd.read_parquet(data_path / "labels.parquet")
    prices_df = pd.read_parquet(data_path / "prices.parquet")

    # Ensure datetime
    for df in [features_df, labels_df, prices_df]:
        df["date"] = pd.to_datetime(df["date"])

    # Ensure prices have required columns
    if "close" not in prices_df.columns and "PX_LAST" in prices_df.columns:
        prices_df["close"] = prices_df["PX_LAST"]
    if "open" not in prices_df.columns:
        prices_df["open"] = prices_df.get("PX_OPEN", prices_df["close"])

    return features_df, labels_df, prices_df


def run_backtest(
    config_path: str,
    model_dir: str,
    data_dir: str,
    output_dir: str,
    start_date: str | None = None,
    end_date: str | None = None,
    walk_forward: bool = False,
) -> dict:
    """
    Run backtest.

    Args:
        config_path: Path to configuration file
        model_dir: Directory with trained models
        data_dir: Directory with processed data
        output_dir: Output directory for results
        start_date: Backtest start date (override config)
        end_date: Backtest end date (override config)
        walk_forward: Run walk-forward optimization

    Returns:
        Backtest results summary
    """
    logger.info("=" * 60)
    logger.info("Starting Backtest")
    logger.info("=" * 60)

    config = load_config(config_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading model...")
    model = load_model(model_dir)
    logger.info(f"Loaded model: {model.model_name}")

    # Load data
    logger.info("Loading data...")
    features_df, labels_df, prices_df = load_data(data_dir)

    logger.info(f"Features: {len(features_df)} records")
    logger.info(f"Prices: {len(prices_df)} records")

    # Get backtest dates
    bt_config = config.get("backtest", {})

    if start_date is None:
        start_date = bt_config.get("start_date", "2023-07-01")
    if end_date is None:
        end_date = bt_config.get("end_date", "2024-12-31")

    logger.info(f"Backtest period: {start_date} to {end_date}")

    # Initialize components
    portfolio_config = config.get("portfolio", {})

    allocator = PortfolioAllocator(
        method=portfolio_config.get("allocation_method", "topk"),
        constraints=PortfolioConstraints(
            max_weight_per_asset=portfolio_config.get("max_weight", 0.1),
            max_leverage=portfolio_config.get("max_leverage", 1.0),
            max_short_weight=0.0,  # Long-only for Korea
        ),
        config=portfolio_config,
    )

    cost_model = TransactionCostModel({
        "commission_bps": bt_config.get("commission_bps", 1.5),
        "tax_bps": bt_config.get("tax_bps", 23.0),
        "spread_bps": bt_config.get("spread_bps", 10.0),
    })

    risk_manager = RiskManager({
        "target_volatility": portfolio_config.get("target_volatility", 0.15),
        "max_drawdown_threshold": portfolio_config.get("max_drawdown", 0.1),
    })

    if walk_forward:
        # Run walk-forward optimization
        logger.info("Running walk-forward optimization...")

        def model_factory():
            # Load fresh model for each fold
            return load_model(model_dir)

        wf_result = run_walk_forward(
            model_factory=model_factory,
            features_df=features_df,
            labels_df=labels_df,
            prices_df=prices_df,
            train_months=bt_config.get("wf_train_months", 24),
            test_months=bt_config.get("wf_test_months", 3),
        )

        # Export walk-forward results
        wf_path = output_path / "walk_forward"
        wf_path.mkdir(exist_ok=True)

        wf_result.combined_returns.to_csv(wf_path / "returns.csv")
        wf_result.fold_metrics.to_csv(wf_path / "fold_metrics.csv", index=False)

        summary = {
            "type": "walk_forward",
            "combined_metrics": wf_result.combined_metrics,
            "n_folds": len(wf_result.folds),
        }

        logger.info("Walk-forward Results:")
        for key, value in wf_result.combined_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")

    else:
        # Run standard backtest
        logger.info("Running backtest...")

        backtest_config = BacktestConfig(
            start_date=pd.Timestamp(start_date),
            end_date=pd.Timestamp(end_date),
            initial_capital=bt_config.get("initial_capital", 100_000_000),
            rebalance_frequency=bt_config.get("rebalance_frequency", "daily"),
            transaction_cost_bps=bt_config.get("total_cost_bps", 35.0),
            slippage_bps=bt_config.get("slippage_bps", 5.0),
        )

        engine = BacktestEngine(
            config=backtest_config,
            allocator=allocator,
            cost_model=cost_model,
            risk_manager=risk_manager,
        )

        result = engine.run(model, features_df, prices_df)

        # Print report
        print_report(result)

        # Export results
        exports = export_report(result, output_path, include_html=True)
        logger.info(f"Exported results to: {output_path}")

        summary = {
            "type": "standard",
            "metrics": result.metrics,
            "n_days": len(result.daily_results),
            "n_trades": len(result.trades_df),
            "output_files": {k: str(v) for k, v in exports.items()},
        }

    # Save summary
    summary["run_date"] = datetime.now().isoformat()
    summary["config_path"] = config_path
    summary["model_dir"] = model_dir
    summary["start_date"] = start_date
    summary["end_date"] = end_date

    summary_path = output_path / "backtest_summary.yaml"
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)

    logger.info("=" * 60)
    logger.info("Backtest Complete")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 60)

    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run backtest")

    parser.add_argument(
        "--config",
        type=str,
        default="config/backtest.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory with trained models",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory with processed data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="Backtest end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward optimization",
    )

    args = parser.parse_args()

    try:
        summary = run_backtest(
            config_path=args.config,
            model_dir=args.model_dir,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            walk_forward=args.walk_forward,
        )

        # Print key metrics
        if "metrics" in summary:
            print("\nKey Metrics:")
            metrics = summary["metrics"]
            print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
