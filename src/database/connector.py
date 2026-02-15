"""
DuckDB Connector

Zero-copy data loading from Parquet files via DuckDB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

# Global connection instance
_connection: duckdb.DuckDBPyConnection | None = None


def get_connection(db_path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """
    Get or create DuckDB connection.

    Args:
        db_path: Path to DuckDB file. None for in-memory.

    Returns:
        DuckDB connection
    """
    global _connection

    if _connection is None:
        if db_path:
            _connection = duckdb.connect(str(db_path))
        else:
            _connection = duckdb.connect()
        logger.info(f"DuckDB connection established: {db_path or 'in-memory'}")

    return _connection


class DuckDBConnector:
    """
    DuckDB connector for efficient Parquet access.

    Features:
        - Zero-copy reads from Parquet
        - SQL query interface
        - Automatic view creation for partitioned data
        - Memory-efficient aggregations
    """

    def __init__(self, db_path: str | Path | None = None):
        """
        Initialize connector.

        Args:
            db_path: Path to DuckDB database file. None for in-memory.
        """
        self.db_path = Path(db_path) if db_path else None
        self.conn = get_connection(db_path)

    def execute(self, query: str, params: list | None = None) -> duckdb.DuckDBPyRelation:
        """Execute SQL query."""
        if params:
            return self.conn.execute(query, params)
        return self.conn.execute(query)

    def query_df(self, query: str, params: list | None = None) -> pd.DataFrame:
        """Execute query and return DataFrame."""
        result = self.execute(query, params)
        return result.df()

    def read_parquet(self, path: str | Path, **kwargs) -> pd.DataFrame:
        """
        Read Parquet file(s) efficiently.

        Args:
            path: Path to Parquet file or glob pattern
            **kwargs: Additional arguments

        Returns:
            DataFrame
        """
        path_str = str(path)
        query = f"SELECT * FROM read_parquet('{path_str}')"
        return self.query_df(query)

    def read_parquet_partitioned(
        self,
        base_path: str | Path,
        partition_cols: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Read partitioned Parquet data.

        Args:
            base_path: Base path containing partitioned data
            partition_cols: Partition column names
            filters: Filters to apply (e.g., {"year": 2024})

        Returns:
            DataFrame
        """
        path_pattern = str(base_path) + "/**/*.parquet"

        query = f"SELECT * FROM read_parquet('{path_pattern}', hive_partitioning=true)"

        if filters:
            conditions = []
            for col, val in filters.items():
                if isinstance(val, str):
                    conditions.append(f"{col} = '{val}'")
                elif isinstance(val, (list, tuple)):
                    val_str = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in val)
                    conditions.append(f"{col} IN ({val_str})")
                else:
                    conditions.append(f"{col} = {val}")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        return self.query_df(query)

    def create_view(self, view_name: str, parquet_path: str | Path) -> None:
        """
        Create a view pointing to Parquet file(s).

        Args:
            view_name: Name of the view
            parquet_path: Path to Parquet file(s)
        """
        path_str = str(parquet_path)
        query = f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT * FROM read_parquet('{path_str}')
        """
        self.execute(query)
        logger.info(f"Created view: {view_name} -> {parquet_path}")

    def create_partitioned_view(
        self,
        view_name: str,
        base_path: str | Path,
    ) -> None:
        """
        Create view for partitioned Parquet data.

        Args:
            view_name: Name of the view
            base_path: Base path with partitioned data
        """
        path_pattern = str(base_path) + "/**/*.parquet"
        query = f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT * FROM read_parquet('{path_pattern}', hive_partitioning=true)
        """
        self.execute(query)
        logger.info(f"Created partitioned view: {view_name}")

    def list_tables(self) -> list[str]:
        """List all tables and views."""
        result = self.query_df("SHOW TABLES")
        return result["name"].tolist() if not result.empty else []

    def table_info(self, table_name: str) -> pd.DataFrame:
        """Get table/view schema information."""
        return self.query_df(f"DESCRIBE {table_name}")

    def get_date_range(self, table_name: str, date_col: str = "date") -> tuple:
        """Get min and max dates from table."""
        result = self.query_df(f"""
            SELECT MIN({date_col}) as min_date, MAX({date_col}) as max_date
            FROM {table_name}
        """)
        return result.iloc[0]["min_date"], result.iloc[0]["max_date"]

    def get_unique_assets(self, table_name: str, asset_col: str = "ticker") -> list[str]:
        """Get unique asset IDs from table."""
        result = self.query_df(f"SELECT DISTINCT {asset_col} FROM {table_name}")
        return result[asset_col].tolist()

    def close(self) -> None:
        """Close connection."""
        global _connection
        if self.conn:
            self.conn.close()
            _connection = None
            logger.info("DuckDB connection closed")


def load_prices(
    conn: DuckDBConnector,
    start_date: str | None = None,
    end_date: str | None = None,
    assets: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convenience function to load price data.

    Args:
        conn: DuckDB connector
        start_date: Start date filter
        end_date: End date filter
        assets: List of asset IDs to load

    Returns:
        Price DataFrame
    """
    conditions = []

    if start_date:
        conditions.append(f"date >= '{start_date}'")
    if end_date:
        conditions.append(f"date <= '{end_date}'")
    if assets:
        asset_str = ", ".join(f"'{a}'" for a in assets)
        conditions.append(f"ticker IN ({asset_str})")

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

    query = f"SELECT * FROM prices{where_clause} ORDER BY date, ticker"

    return conn.query_df(query)


def load_features(
    conn: DuckDBConnector,
    start_date: str | None = None,
    end_date: str | None = None,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convenience function to load features.

    Args:
        conn: DuckDB connector
        start_date: Start date filter
        end_date: End date filter
        feature_cols: Specific features to load

    Returns:
        Features DataFrame
    """
    if feature_cols:
        cols = ", ".join(["date", "ticker"] + feature_cols)
    else:
        cols = "*"

    conditions = []
    if start_date:
        conditions.append(f"date >= '{start_date}'")
    if end_date:
        conditions.append(f"date <= '{end_date}'")

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

    query = f"SELECT {cols} FROM features{where_clause} ORDER BY date, ticker"

    return conn.query_df(query)
