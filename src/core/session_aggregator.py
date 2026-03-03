"""
Implements the MapReduce pattern for aggregating session data into
user-level features and targets for orbital segmentation.
"""

from types import NoneType
from typing import List, Callable
from collections.abc import Hashable
from dataclasses import dataclass, field

import pandas as pd
from datetime import datetime

from src.data.session_processor import BaseConfig, SessionProcessor

# ParseSessionType = Callable[[pd.Series], Dict[str, float]]


@dataclass
class MapReduceRule:
    """Defines how to apply and aggregate data using specific function."""
    map_func: str | Callable  # Column name or function
    reduce_func: str  # 'sum', 'max', 'min', 'mean', 'count'
    alias: str | NoneType = None

    def __post_init__(self):
        if isinstance(self.map_func, str):
            column_name = self.map_func
            self.map_func = lambda row: row[column_name]
            self.alias = self.alias or column_name


@dataclass
class AggregationRule:
    freq: str = 'D' # granularity (M/W/D/H)
    periods: int = 30
    parse_sessions: List[MapReduceRule] = None
    parse_discrete_events: List[MapReduceRule] = None


@dataclass
class SessionAggregatorConfig(BaseConfig):
    parse_sessions: List[MapReduceRule] = field(default_factory=list)
    parse_discrete_events: List[MapReduceRule] = field(default_factory=list)
    freq: str = field(default='D')
    periods: int = field(default=30)

    start: datetime | str = field(default=None)
    end: datetime | str = field(default=None)
    window: List[datetime] = field(default_factory=list)


class SessionAggregator:
    """
    Aggregates raw sessions.
    """
    session_processor: SessionProcessor
    agg_rule: AggregationRule
    start: datetime
    window: List[datetime]
    end: datetime
    config: BaseConfig
    sessions_df: pd.DataFrame = None
    discrete_events_df: pd.DataFrame = None
    user_stats_df: pd.DataFrame = None

    def __init__(
        self,
        session_processor: SessionProcessor,
        agg_rule: AggregationRule
    ) -> None:
        """
        Initialize session aggregator.
        
        Args:
            ...
            # freq: Time frequency for window (M=month, W=week, D=day, H=hour)
            # periods: ...
            # parse_session: Rules for session-level aggregation
            # parse_discrete_events: Rules for user-level aggregation
        """
        self.config = session_processor.config
        self.session_processor = session_processor
        self.agg_rule = agg_rule

    def set_time_window(self, start: datetime) -> None:
        self.start = start
        self.window = [
            self.start + pd.Timedelta(i, self.agg_rule.freq)
            for i in range(self.agg_rule.periods)
        ]
        self.end = self.start + pd.Timedelta(self.agg_rule.periods, self.agg_rule.freq)
        

    def apply_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        # Create time periods: { (w_{i-1}, w_{i}] }
        # df['gtd'] = self.agg_rule.freq # TODO: ???
        df['period'] = pd.cut(df[self.config.timestamp_col], bins=self.window + [self.end], labels=False) + 1

        for rule in self.agg_rule.parse_sessions:
            df[rule.alias] = df.apply(rule.map_func, axis=1)
        df =  df[
            [self.session_processor.user_id_col, 'period'] \
            + [rule.alias for rule in self.agg_rule.parse_sessions]
        ]

        return df

    def aggregate_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        reduced_df = (
            self.apply_sessions(df)
            .groupby([self.session_processor.user_id_col, 'period'], as_index=False)
            .agg({rule.alias: rule.reduce_func for rule in self.agg_rule.parse_sessions})
        )
        return reduced_df

    def apply_events(self, df: pd.DataFrame) -> pd.DataFrame:

        for rule in self.agg_rule.parse_discrete_events:
            df[rule.alias] = df.apply(rule.map_func, axis=1)

        df =  df[
            [self.config.user_id_col] \
            + [rule.alias for rule in self.agg_rule.parse_discrete_events]
        ]

        return df

    def aggregate_events(self, df: pd.DataFrame) -> pd.DataFrame:
        reduced_df = (
            self.apply_events(df)
            .groupby([self.config.user_id_col], as_index=False)
            .agg({rule.alias: rule.reduce_func for rule in self.agg_rule.parse_discrete_events})
        )
        return reduced_df

    def aggregate(
            self, 
            start: datetime,
            users: List[Hashable] | NoneType = None,
        ) -> pd.DataFrame:
        """
        start: Start time for aggregation window
        # periods: Number of time periods to create
        """
        self.set_time_window(start)
        # filter by (start, end] + users (optional)
        self.sessions_df = self.session_processor.filter(start=self.start, end=self.end, users=users)
        # aggregate by discrete events
        self.discrete_events_df = self.aggregate_sessions(self.sessions_df)
        # aggregate by users
        self.user_stats_df = self.aggregate_events(self.discrete_events_df)

        return self.user_stats_df
