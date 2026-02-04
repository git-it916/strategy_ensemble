"""
Database Module

DuckDB connection and schema management for efficient data access.
"""

from .connector import DuckDBConnector, get_connection
from .schema_manager import SchemaManager

__all__ = ["DuckDBConnector", "get_connection", "SchemaManager"]
