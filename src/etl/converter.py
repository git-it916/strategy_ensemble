"""
Parquet Converter

Convert CSV data to compressed, partitioned Parquet format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class ParquetConverter:
    """
    Convert and partition data to Parquet format.

    Features:
        - Snappy compression
        - Year/month partitioning
        - Schema enforcement
        - Incremental updates
    """

    def __init__(
        self,
        output_dir: str | Path,
        compression: str = "snappy",
        partition_cols: list[str] | None = None,
    ):
        """
        Initialize converter.

        Args:
            output_dir: Output directory for Parquet files
            compression: Compression codec (snappy, gzip, zstd)
            partition_cols: Columns to partition by
        """
        self.output_dir = Path(output_dir)
        self.compression = compression
        self.partition_cols = partition_cols or []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
        partition: bool = False,
    ) -> Path:
        """
        Convert DataFrame to Parquet.

        Args:
            df: Input DataFrame
            name: Output file name (without extension)
            partition: Whether to partition the data

        Returns:
            Output path
        """
        if partition and self.partition_cols:
            return self._write_partitioned(df, name)
        else:
            return self._write_single(df, name)

    def _write_single(self, df: pd.DataFrame, name: str) -> Path:
        """Write single Parquet file."""
        output_path = self.output_dir / f"{name}.parquet"

        # Convert to PyArrow table
        table = pa.Table.from_pandas(df)

        # Write with compression
        pq.write_table(
            table,
            output_path,
            compression=self.compression,
        )

        logger.info(f"Wrote {len(df)} rows to {output_path}")
        return output_path

    def _write_partitioned(self, df: pd.DataFrame, name: str) -> Path:
        """Write partitioned Parquet dataset."""
        output_path = self.output_dir / f"{name}_partitioned"
        output_path.mkdir(exist_ok=True)

        # Ensure partition columns exist
        if "date" in df.columns and "year" not in df.columns:
            df = df.copy()
            df["year"] = pd.to_datetime(df["date"]).dt.year
            df["month"] = pd.to_datetime(df["date"]).dt.month

        # Get partition columns that exist
        available_partitions = [c for c in self.partition_cols if c in df.columns]

        if not available_partitions:
            logger.warning("No partition columns available, writing single file")
            return self._write_single(df, name)

        # Convert to PyArrow
        table = pa.Table.from_pandas(df)

        # Write partitioned dataset
        pq.write_to_dataset(
            table,
            output_path,
            partition_cols=available_partitions,
            compression=self.compression,
        )

        logger.info(f"Wrote partitioned dataset to {output_path}")
        return output_path

    def convert_csv(
        self,
        csv_path: str | Path,
        name: str,
        date_cols: list[str] | None = None,
        dtype: dict | None = None,
        partition: bool = False,
    ) -> Path:
        """
        Convert CSV file to Parquet.

        Args:
            csv_path: Path to CSV file
            name: Output name
            date_cols: Columns to parse as dates
            dtype: Column data types
            partition: Whether to partition

        Returns:
            Output path
        """
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        # Read CSV
        df = pd.read_csv(
            csv_path,
            parse_dates=date_cols,
            dtype=dtype,
        )

        logger.info(f"Read {len(df)} rows from {csv_path}")

        return self.convert_dataframe(df, name, partition)

    def append_data(
        self,
        df: pd.DataFrame,
        name: str,
    ) -> Path:
        """
        Append new data to existing Parquet.

        Args:
            df: New data to append
            name: Dataset name

        Returns:
            Output path
        """
        existing_path = self.output_dir / f"{name}.parquet"

        if existing_path.exists():
            # Read existing
            existing_df = pd.read_parquet(existing_path)

            # Combine
            combined = pd.concat([existing_df, df], ignore_index=True)

            # Remove duplicates based on date + ticker
            if "date" in combined.columns and "ticker" in combined.columns:
                combined = combined.drop_duplicates(
                    subset=["date", "ticker"],
                    keep="last"
                )

            df = combined
            logger.info(f"Appended to existing data, total rows: {len(df)}")

        return self._write_single(df, name)


def convert_csv_to_parquet(
    csv_path: str | Path,
    output_dir: str | Path,
    name: str | None = None,
) -> Path:
    """
    Convenience function to convert CSV to Parquet.

    Args:
        csv_path: Input CSV path
        output_dir: Output directory
        name: Output name (default: CSV filename)

    Returns:
        Output path
    """
    csv_path = Path(csv_path)
    name = name or csv_path.stem

    converter = ParquetConverter(output_dir)
    return converter.convert_csv(csv_path, name)


def convert_bloomberg_export(
    csv_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Convert Bloomberg CSV export to Parquet.

    Handles typical Bloomberg export format with multiple securities.

    Args:
        csv_path: Path to Bloomberg CSV
        output_dir: Output directory

    Returns:
        Dict of dataset name -> path
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read Bloomberg CSV (typically has specific format)
    df = pd.read_csv(csv_path)

    # Common Bloomberg column mappings
    column_map = {
        "Date": "date",
        "DATE": "date",
        "Ticker": "ticker",
        "TICKER": "ticker",
        "PX_LAST": "close",
        "PX_OPEN": "open",
        "PX_HIGH": "high",
        "PX_LOW": "low",
        "PX_VOLUME": "volume",
    }

    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    converter = ParquetConverter(output_dir)
    output_path = converter.convert_dataframe(df, "prices")

    return {"prices": output_path}
