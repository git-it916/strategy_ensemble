"""
Data Validation Module

Provides data quality checks and validation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..common import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status}: {self.name} - {self.message}"


@dataclass
class ValidationReport:
    """Complete validation report for a dataset."""

    results: list[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Check if all validations passed."""
        return all(r.passed for r in self.results)

    @property
    def n_passed(self) -> int:
        """Number of passed checks."""
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        """Number of failed checks."""
        return sum(1 for r in self.results if not r.passed)

    def add(self, result: ValidationResult) -> None:
        """Add a validation result."""
        self.results.append(result)

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"Validation Report: {self.n_passed}/{len(self.results)} checks passed",
            "=" * 60,
        ]
        for r in self.results:
            lines.append(str(r))
        lines.append("=" * 60)
        return "\n".join(lines)


class DataValidator:
    """
    Data quality validator for market data.

    Performs various checks on price, fundamental, and feature data.
    """

    def __init__(
        self,
        min_price: float = 100,
        max_price: float = 10_000_000,
        max_return: float = 0.5,
        min_volume: int = 0,
        max_missing_pct: float = 0.1,
    ):
        """
        Initialize validator with thresholds.

        Args:
            min_price: Minimum valid price
            max_price: Maximum valid price
            max_return: Maximum single-day return (absolute)
            min_volume: Minimum valid volume
            max_missing_pct: Maximum allowed missing percentage
        """
        self.min_price = min_price
        self.max_price = max_price
        self.max_return = max_return
        self.min_volume = min_volume
        self.max_missing_pct = max_missing_pct

    def validate_price_data(self, df: pd.DataFrame) -> ValidationReport:
        """
        Validate price data quality.

        Args:
            df: Price DataFrame with columns [date, asset_id, close, volume, ...]

        Returns:
            ValidationReport with all check results
        """
        report = ValidationReport()

        # Check required columns
        report.add(self._check_required_columns(
            df, ["date", "asset_id", "close"], "price data"
        ))

        # Check date validity
        if "date" in df.columns:
            report.add(self._check_date_column(df))

        # Check price range
        if "close" in df.columns:
            report.add(self._check_price_range(df, "close"))

        # Check for negative prices
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                report.add(self._check_non_negative(df, col))

        # Check OHLC consistency
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            report.add(self._check_ohlc_consistency(df))

        # Check volume
        if "volume" in df.columns:
            report.add(self._check_volume(df))

        # Check for excessive returns
        if "close" in df.columns:
            report.add(self._check_returns(df))

        # Check for duplicates
        report.add(self._check_duplicates(df, ["date", "asset_id"]))

        # Check missing values
        report.add(self._check_missing_values(df))

        return report

    def validate_fundamental_data(self, df: pd.DataFrame) -> ValidationReport:
        """
        Validate fundamental data quality.

        Args:
            df: Fundamental DataFrame with columns [asset_id, pbr, per, ...]

        Returns:
            ValidationReport with all check results
        """
        report = ValidationReport()

        # Check required columns
        report.add(self._check_required_columns(df, ["asset_id"], "fundamental data"))

        # Check for negative ratios where they shouldn't be
        for col in ["pbr", "per"]:
            if col in df.columns:
                report.add(self._check_ratio_range(df, col))

        # Check ROE range
        if "roe" in df.columns:
            report.add(self._check_roe_range(df))

        # Check for duplicates
        if "date" in df.columns:
            report.add(self._check_duplicates(df, ["date", "asset_id"]))
        else:
            report.add(self._check_duplicates(df, ["asset_id"]))

        # Check missing values
        report.add(self._check_missing_values(df))

        return report

    def validate_features_df(self, df: pd.DataFrame) -> ValidationReport:
        """
        Validate features DataFrame.

        Args:
            df: Features DataFrame

        Returns:
            ValidationReport with all check results
        """
        report = ValidationReport()

        # Check required columns
        report.add(self._check_required_columns(df, ["date", "asset_id"], "features"))

        # Check date column
        if "date" in df.columns:
            report.add(self._check_date_column(df))

        # Check for infinite values
        report.add(self._check_infinite_values(df))

        # Check for duplicates
        report.add(self._check_duplicates(df, ["date", "asset_id"]))

        # Check missing values
        report.add(self._check_missing_values(df))

        # Check feature ranges (z-scored features should be reasonable)
        report.add(self._check_feature_ranges(df))

        return report

    def validate_labels_df(self, df: pd.DataFrame) -> ValidationReport:
        """
        Validate labels DataFrame.

        Args:
            df: Labels DataFrame

        Returns:
            ValidationReport with all check results
        """
        report = ValidationReport()

        # Check required columns
        report.add(self._check_required_columns(
            df, ["date", "asset_id", "y_reg"], "labels"
        ))

        # Check date column
        if "date" in df.columns:
            report.add(self._check_date_column(df))

        # Check label range
        if "y_reg" in df.columns:
            report.add(self._check_label_range(df))

        # Check classification labels
        if "y_cls" in df.columns:
            report.add(self._check_classification_labels(df))

        # Check for duplicates
        report.add(self._check_duplicates(df, ["date", "asset_id"]))

        return report

    # ==========================================================================
    # Individual Check Methods
    # ==========================================================================

    def _check_required_columns(
        self,
        df: pd.DataFrame,
        required: list[str],
        data_name: str,
    ) -> ValidationResult:
        """Check if required columns are present."""
        missing = [c for c in required if c not in df.columns]

        if missing:
            return ValidationResult(
                name="required_columns",
                passed=False,
                message=f"Missing columns in {data_name}: {missing}",
                details={"missing": missing, "found": df.columns.tolist()},
            )

        return ValidationResult(
            name="required_columns",
            passed=True,
            message=f"All required columns present in {data_name}",
        )

    def _check_date_column(self, df: pd.DataFrame) -> ValidationResult:
        """Check date column validity."""
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            return ValidationResult(
                name="date_type",
                passed=False,
                message="Date column is not datetime type",
                details={"dtype": str(df["date"].dtype)},
            )

        # Check for NaT
        nat_count = df["date"].isna().sum()
        if nat_count > 0:
            return ValidationResult(
                name="date_type",
                passed=False,
                message=f"Date column contains {nat_count} NaT values",
            )

        # Check date range
        min_date = df["date"].min()
        max_date = df["date"].max()

        return ValidationResult(
            name="date_type",
            passed=True,
            message=f"Date range: {min_date.date()} to {max_date.date()}",
            details={"min_date": str(min_date), "max_date": str(max_date)},
        )

    def _check_price_range(self, df: pd.DataFrame, col: str) -> ValidationResult:
        """Check if prices are within valid range."""
        below_min = (df[col] < self.min_price).sum()
        above_max = (df[col] > self.max_price).sum()
        invalid = below_min + above_max

        if invalid > 0:
            return ValidationResult(
                name=f"price_range_{col}",
                passed=False,
                message=f"{invalid} prices outside valid range [{self.min_price}, {self.max_price}]",
                details={"below_min": below_min, "above_max": above_max},
            )

        return ValidationResult(
            name=f"price_range_{col}",
            passed=True,
            message=f"All prices in {col} within valid range",
        )

    def _check_non_negative(self, df: pd.DataFrame, col: str) -> ValidationResult:
        """Check for negative values."""
        negative = (df[col] < 0).sum()

        if negative > 0:
            return ValidationResult(
                name=f"non_negative_{col}",
                passed=False,
                message=f"{negative} negative values in {col}",
            )

        return ValidationResult(
            name=f"non_negative_{col}",
            passed=True,
            message=f"No negative values in {col}",
        )

    def _check_ohlc_consistency(self, df: pd.DataFrame) -> ValidationResult:
        """Check OHLC price consistency (High >= Low, etc.)."""
        issues = []

        # High should be >= Low
        high_low = (df["high"] < df["low"]).sum()
        if high_low > 0:
            issues.append(f"{high_low} rows where high < low")

        # High should be >= Open, Close
        high_open = (df["high"] < df["open"]).sum()
        high_close = (df["high"] < df["close"]).sum()
        if high_open > 0:
            issues.append(f"{high_open} rows where high < open")
        if high_close > 0:
            issues.append(f"{high_close} rows where high < close")

        # Low should be <= Open, Close
        low_open = (df["low"] > df["open"]).sum()
        low_close = (df["low"] > df["close"]).sum()
        if low_open > 0:
            issues.append(f"{low_open} rows where low > open")
        if low_close > 0:
            issues.append(f"{low_close} rows where low > close")

        if issues:
            return ValidationResult(
                name="ohlc_consistency",
                passed=False,
                message="; ".join(issues),
            )

        return ValidationResult(
            name="ohlc_consistency",
            passed=True,
            message="OHLC prices are consistent",
        )

    def _check_volume(self, df: pd.DataFrame) -> ValidationResult:
        """Check volume validity."""
        negative = (df["volume"] < 0).sum()
        zero = (df["volume"] == 0).sum()

        if negative > 0:
            return ValidationResult(
                name="volume",
                passed=False,
                message=f"{negative} negative volume values",
            )

        details = {"zero_volume_count": zero, "zero_volume_pct": zero / len(df)}

        return ValidationResult(
            name="volume",
            passed=True,
            message=f"Volume valid ({zero} zero-volume days)",
            details=details,
        )

    def _check_returns(self, df: pd.DataFrame) -> ValidationResult:
        """Check for excessive single-day returns."""
        # Calculate returns by asset
        df_sorted = df.sort_values(["asset_id", "date"])
        df_sorted["return"] = df_sorted.groupby("asset_id")["close"].pct_change()

        excessive = (df_sorted["return"].abs() > self.max_return).sum()

        if excessive > 0:
            return ValidationResult(
                name="returns",
                passed=False,
                message=f"{excessive} excessive returns (>{self.max_return*100}%)",
                details={"threshold": self.max_return},
            )

        return ValidationResult(
            name="returns",
            passed=True,
            message="No excessive single-day returns",
        )

    def _check_duplicates(
        self,
        df: pd.DataFrame,
        subset: list[str],
    ) -> ValidationResult:
        """Check for duplicate rows."""
        # Only check columns that exist
        subset = [c for c in subset if c in df.columns]

        if not subset:
            return ValidationResult(
                name="duplicates",
                passed=True,
                message="No duplicate check columns found",
            )

        duplicates = df.duplicated(subset=subset).sum()

        if duplicates > 0:
            return ValidationResult(
                name="duplicates",
                passed=False,
                message=f"{duplicates} duplicate rows found",
                details={"subset": subset},
            )

        return ValidationResult(
            name="duplicates",
            passed=True,
            message="No duplicate rows",
        )

    def _check_missing_values(self, df: pd.DataFrame) -> ValidationResult:
        """Check for excessive missing values."""
        missing_pct = df.isna().mean()
        high_missing = missing_pct[missing_pct > self.max_missing_pct]

        if len(high_missing) > 0:
            return ValidationResult(
                name="missing_values",
                passed=False,
                message=f"{len(high_missing)} columns with >{self.max_missing_pct*100}% missing",
                details={"columns": high_missing.to_dict()},
            )

        total_missing = df.isna().sum().sum()
        total_cells = df.size

        return ValidationResult(
            name="missing_values",
            passed=True,
            message=f"Missing: {total_missing}/{total_cells} ({total_missing/total_cells*100:.2f}%)",
        )

    def _check_infinite_values(self, df: pd.DataFrame) -> ValidationResult:
        """Check for infinite values."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        inf_count = 0
        inf_cols = []

        for col in numeric_cols:
            col_inf = np.isinf(df[col]).sum()
            if col_inf > 0:
                inf_count += col_inf
                inf_cols.append(col)

        if inf_count > 0:
            return ValidationResult(
                name="infinite_values",
                passed=False,
                message=f"{inf_count} infinite values in columns: {inf_cols}",
            )

        return ValidationResult(
            name="infinite_values",
            passed=True,
            message="No infinite values",
        )

    def _check_feature_ranges(self, df: pd.DataFrame) -> ValidationResult:
        """Check feature value ranges."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude = ["date", "asset_id"]
        feature_cols = [c for c in numeric_cols if c not in exclude]

        extreme_cols = []
        for col in feature_cols:
            # Check if values are unreasonably large (for z-scored features)
            max_abs = df[col].abs().max()
            if max_abs > 100:  # Extremely large z-score
                extreme_cols.append((col, max_abs))

        if extreme_cols:
            return ValidationResult(
                name="feature_ranges",
                passed=False,
                message=f"Extreme values in {len(extreme_cols)} columns",
                details={"columns": extreme_cols[:10]},  # Show first 10
            )

        return ValidationResult(
            name="feature_ranges",
            passed=True,
            message="Feature ranges are reasonable",
        )

    def _check_ratio_range(self, df: pd.DataFrame, col: str) -> ValidationResult:
        """Check ratio range (e.g., PBR, PER)."""
        # These ratios can be negative in some cases but extremely negative is suspicious
        extreme_negative = (df[col] < -10).sum()
        extreme_positive = (df[col] > 1000).sum()

        if extreme_negative > 0 or extreme_positive > 0:
            return ValidationResult(
                name=f"ratio_range_{col}",
                passed=False,
                message=f"Extreme {col} values: {extreme_negative} < -10, {extreme_positive} > 1000",
            )

        return ValidationResult(
            name=f"ratio_range_{col}",
            passed=True,
            message=f"{col} values in reasonable range",
        )

    def _check_roe_range(self, df: pd.DataFrame) -> ValidationResult:
        """Check ROE range."""
        # ROE typically between -1 and 1 (or -100% to 100%)
        outside = ((df["roe"] < -1) | (df["roe"] > 1)).sum()

        # If stored as percentage
        if df["roe"].abs().max() > 1:
            outside = ((df["roe"] < -100) | (df["roe"] > 100)).sum()

        if outside > 0:
            return ValidationResult(
                name="roe_range",
                passed=False,
                message=f"{outside} ROE values outside typical range",
            )

        return ValidationResult(
            name="roe_range",
            passed=True,
            message="ROE values in reasonable range",
        )

    def _check_label_range(self, df: pd.DataFrame) -> ValidationResult:
        """Check label (forward return) range."""
        # Forward returns should typically be within reasonable bounds
        extreme = ((df["y_reg"] < -0.9) | (df["y_reg"] > 5)).sum()

        if extreme > 0:
            return ValidationResult(
                name="label_range",
                passed=False,
                message=f"{extreme} labels with extreme values",
            )

        return ValidationResult(
            name="label_range",
            passed=True,
            message="Label values in reasonable range",
        )

    def _check_classification_labels(self, df: pd.DataFrame) -> ValidationResult:
        """Check classification label validity."""
        unique_labels = df["y_cls"].unique()

        # Check if binary or multi-class
        if not all(isinstance(x, (int, np.integer)) for x in unique_labels if pd.notna(x)):
            return ValidationResult(
                name="classification_labels",
                passed=False,
                message="Classification labels should be integers",
            )

        return ValidationResult(
            name="classification_labels",
            passed=True,
            message=f"Classification labels valid: {sorted(unique_labels)}",
        )


def validate_data(
    df: pd.DataFrame,
    data_type: str = "price",
    strict: bool = False,
) -> ValidationReport:
    """
    Convenience function to validate data.

    Args:
        df: DataFrame to validate
        data_type: Type of data ('price', 'fundamental', 'features', 'labels')
        strict: If True, raise exception on validation failure

    Returns:
        ValidationReport

    Raises:
        ValueError: If strict=True and validation fails
    """
    validator = DataValidator()

    if data_type == "price":
        report = validator.validate_price_data(df)
    elif data_type == "fundamental":
        report = validator.validate_fundamental_data(df)
    elif data_type == "features":
        report = validator.validate_features_df(df)
    elif data_type == "labels":
        report = validator.validate_labels_df(df)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    if strict and not report.passed:
        logger.error(str(report))
        raise ValueError(f"Data validation failed: {report.n_failed} checks failed")

    logger.info(str(report))
    return report
