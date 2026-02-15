"""
Schema Manager

Manage DuckDB views and partitioned data structure.
"""

from __future__ import annotations

from pathlib import Path
import logging

from .connector import DuckDBConnector

logger = logging.getLogger(__name__)


class SchemaManager:
    """
    Manage database schema and views.

    Responsibilities:
        - Create views for Parquet data
        - Manage data partitioning
        - Schema validation
    """

    def __init__(self, connector: DuckDBConnector, data_dir: str | Path):
        """
        Initialize schema manager.

        Args:
            connector: DuckDB connector
            data_dir: Base data directory
        """
        self.conn = connector
        self.data_dir = Path(data_dir)

    def setup_views(self) -> None:
        """Set up all standard views."""
        processed_dir = self.data_dir / "processed"

        # Core data views
        view_configs = [
            ("prices", processed_dir / "prices.parquet"),
            ("features", processed_dir / "features.parquet"),
            ("labels", processed_dir / "labels.parquet"),
        ]

        for view_name, parquet_path in view_configs:
            if parquet_path.exists():
                self.conn.create_view(view_name, parquet_path)
                logger.info(f"Created view: {view_name}")
            else:
                logger.warning(f"Parquet not found: {parquet_path}")

        # Check for partitioned data
        partitioned_dirs = [
            ("prices_hist", processed_dir / "prices_partitioned"),
            ("features_hist", processed_dir / "features_partitioned"),
        ]

        for view_name, base_path in partitioned_dirs:
            if base_path.exists():
                self.conn.create_partitioned_view(view_name, base_path)
                logger.info(f"Created partitioned view: {view_name}")

    def create_price_view(self, path: str | Path) -> None:
        """Create prices view."""
        self.conn.create_view("prices", path)

    def create_feature_view(self, path: str | Path) -> None:
        """Create features view."""
        self.conn.create_view("features", path)

    def create_label_view(self, path: str | Path) -> None:
        """Create labels view."""
        self.conn.create_view("labels", path)

    def validate_schema(self) -> dict[str, bool]:
        """
        Validate that required tables/views exist.

        Returns:
            Dict of table_name -> exists
        """
        required = ["prices", "features", "labels"]
        existing = set(self.conn.list_tables())

        return {name: name in existing for name in required}

    def get_data_summary(self) -> dict:
        """
        Get summary of available data.

        Returns:
            Summary dict with date ranges and counts
        """
        summary = {}

        for table in ["prices", "features", "labels"]:
            if table not in self.conn.list_tables():
                continue

            try:
                count = self.conn.query_df(f"SELECT COUNT(*) as cnt FROM {table}").iloc[0]["cnt"]
                min_date, max_date = self.conn.get_date_range(table)
                n_assets = len(self.conn.get_unique_assets(table))

                summary[table] = {
                    "count": count,
                    "min_date": str(min_date),
                    "max_date": str(max_date),
                    "n_assets": n_assets,
                }
            except Exception as e:
                logger.warning(f"Error getting summary for {table}: {e}")

        return summary

    def create_daily_returns_view(self) -> None:
        """Create view for daily returns calculation."""
        query = """
            CREATE OR REPLACE VIEW daily_returns AS
            SELECT
                date,
                ticker,
                close,
                close / LAG(close) OVER (PARTITION BY ticker ORDER BY date) - 1 as daily_return
            FROM prices
            ORDER BY date, ticker
        """
        self.conn.execute(query)
        logger.info("Created daily_returns view")

    def create_feature_stats_view(self) -> None:
        """Create view for feature statistics."""
        query = """
            CREATE OR REPLACE VIEW feature_stats AS
            SELECT
                date,
                COUNT(*) as n_assets,
                AVG(ret_5d) as avg_ret_5d,
                AVG(ret_21d) as avg_ret_21d,
                AVG(vol_21d) as avg_vol_21d,
                AVG(rsi_14) as avg_rsi
            FROM features
            GROUP BY date
            ORDER BY date
        """
        self.conn.execute(query)
        logger.info("Created feature_stats view")


def initialize_database(
    db_path: str | Path,
    data_dir: str | Path,
) -> tuple[DuckDBConnector, SchemaManager]:
    """
    Initialize database with all views.

    Args:
        db_path: Path to DuckDB file
        data_dir: Base data directory

    Returns:
        Tuple of (connector, schema_manager)
    """
    connector = DuckDBConnector(db_path)
    schema_mgr = SchemaManager(connector, data_dir)
    schema_mgr.setup_views()

    # Create additional analytical views
    try:
        schema_mgr.create_daily_returns_view()
        schema_mgr.create_feature_stats_view()
    except Exception as e:
        logger.warning(f"Could not create analytical views: {e}")

    return connector, schema_mgr
