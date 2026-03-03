
from types import NoneType
from typing import Dict, Any, List, Union
from collections.abc import Hashable

import pandas as pd
from datetime import datetime

from dataclasses import dataclass, field

@dataclass
class BaseConfig:
    """Configuration for session processor."""
    timestamp_col: str
    user_id_col: str
    required_columns: List[str] = field(default_factory=list)   


class SessionProcessor:
    """
    Processes raw session data for orbital segmentation framework.
    
    Handles data loading, preprocessing, and filtering of session data
    to prepare for feature extraction and orbital assignment.

    # Each row in the input DataFrame represents a single session.
    """
    timestamp_col: str
    user_id_col: str
    required_columns: List[str]
    config: BaseConfig
    df: pd.DataFrame
    
    def __init__(
            self,
            timestamp_col: str,
            user_id_col: str,
            required_columns: List[str] = []
        ) -> None:
        self.timestamp_col = timestamp_col
        self.user_id_col = user_id_col
        self.required_columns = required_columns
        self.config = BaseConfig(
            timestamp_col = timestamp_col,
            user_id_col = user_id_col,
            required_columns = required_columns
        )


    def load(self, file_path: str) -> pd.DataFrame:
        """
        Load session data from file.
        
        Args:
            file_path: Path to the data file (CSV, parquet)
            
        Returns:
            DataFrame with session data
        """
        if file_path.endswith('.csv'):
            self.df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            self.df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        self.df = self.preprocess(self.df)
        return self.df

    def preprocess(self, sessions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess session data for analysis.
        
        Args:
            sessions_df: Raw session DataFrame
            
        Returns:
            Preprocessed Session DataFrame
        """
        df = sessions_df.copy()
        
        # Convert timestamp to datetime
        if self.timestamp_col in df.columns:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        
        # Ensure required columns exist
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort by timestamp
        df = df.sort_values(self.timestamp_col)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        return df
    
    def filter(
        self, 
        # sessions_df: pd.DataFrame,
        start: datetime | str | NoneType = None,
        end: datetime | str | NoneType = None,
        users: List[Hashable] | NoneType = None
        ) -> pd.DataFrame:
        """
        Filter sessions by time range and users.
        
        Args:
            sessions_df: Session DataFrame to filter
            start: Start timestamp (inclusive)
            end: End timestamp (exclusive)
            users: List of user IDs to include
            
        Returns:
            Filtered Session DataFrame
        """
        df = self.df.copy()
        
        # Filter by (start, end]
        if start is not None:
            start = pd.to_datetime(start)
            df = df[df[self.timestamp_col] > start]
        
        if end is not None:
            end = pd.to_datetime(end)
            df = df[df[self.timestamp_col] <= end]
        
        # Filter by users
        if users is not None:
            df = df[df[self.user_id_col].isin(users)]
        
        return df
    
    def get_users_power(self):
        return self.df[self.user_id_col].nunique()
    
    def get_users_sample(self):
        return self.df[self.user_id_col].unique()
