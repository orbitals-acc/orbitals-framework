"""
Feature extraction pipeline for orbital segmentation framework ?
"""
from typing import List, Hashable, Tuple, Optional, Dict
from dataclasses import dataclass, field

from datetime import datetime
import pandas as pd

from src.data.session_processor import BaseConfig, SessionProcessor
from src.core.session_aggregator import MapReduceRule, AggregationRule, SessionAggregator


@dataclass
class FeatureDataStore:
    session_aggregator: SessionAggregator
    agg_rule: AggregationRule
    data: pd.DataFrame
    mode: str = ''


class FeatureExtractor:
    session_processor: SessionProcessor
    feature_rules: List[AggregationRule]
    target_rule: AggregationRule
    feature_extractions: List[FeatureDataStore] = []
    target_extraction: Optional[FeatureDataStore] = None

    def __init__(
            self,
            session_processor: SessionProcessor,
            feature_rules: List[AggregationRule],
            target_rule: AggregationRule
        ) -> None:
        self.session_processor = session_processor
        self.feature_rules = feature_rules
        self.target_rule = target_rule
        # self.extractions = []

    def clear(self):
        self.feature_extractions = []
        self.target_extraction = None
    
    def extract_by(
        self,
        agg_rule: AggregationRule,
        timestamps: List[datetime],
        users: Optional[List[Hashable]] = None,
        mode: str = '',
     ) -> FeatureDataStore:

     session_aggregator = SessionAggregator(
        session_processor = self.session_processor,
        agg_rule=agg_rule,
    )
     
     if users is None:
        users = self.session_processor.get_users_sample()
     
     tables = []
     for timestamp in timestamps:

        start_ts = timestamp
        if mode != 'target':
            start_ts -= pd.Timedelta(agg_rule.periods, agg_rule.freq)

        user_df = session_aggregator.aggregate(start=start_ts, users=users)
        user_df = user_df.set_index(self.session_processor.config.user_id_col)
        user_df = user_df.reindex(users, level=1)
        user_df = user_df.reset_index()

        tables.append(user_df.assign(timestamp=timestamp))
    
     data = pd.concat([
         table.set_index(['timestamp', self.session_processor.config.user_id_col])
         for table in tables
     ], axis = 0)

     data.columns = [col + f"_{mode}" for col in data.columns]
    
     return FeatureDataStore(
        session_aggregator=session_aggregator,
        agg_rule=agg_rule,
        data=data,
        mode=mode,
     )

    def extract_features(self,
            timestamps: List[datetime],
            users: Optional[List[Hashable]] = None
        ) -> None:

        for idx, feature_rule in enumerate(self.feature_rules):
            self.feature_extractions.append(
                self.extract_by(
                    agg_rule=feature_rule,
                    timestamps=timestamps,
                    users=users,
                    mode=f"{idx}"
                )
            )

        return

    def extract_target(self,
            timestamps: List[datetime],
            users: Optional[List[Hashable]] = None
        ) -> None:
        
        self.target_extraction = self.extract_by(
            agg_rule=self.target_rule,
            timestamps=timestamps,
            users=users,
            mode='target',
        )

        return

    def extract_data(
            self,
            timestamps: List[datetime],
            users: Optional[List[Hashable]] = None,
            with_target=True
        ) -> pd.DataFrame:

        if not self.feature_extractions:
            self.extract_features(timestamps, users)

        tables = [
            store.data for store in self.feature_extractions
        ]

        if with_target:
            if self.target_extraction is None:
                self.extract_target(timestamps, users)
            tables.append(self.target_extraction.data)

        return pd.concat(tables, axis=1)
    
    def read_data(self, file_path: str) -> pd.DataFrame:

        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            data = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        config = self.session_processor.config
        data[config.timestamp_col] = pd.to_datetime(data[config.timestamp_col])	
        data.set_index([config.timestamp_col, config.user_id_col], inplace=True)

        return data

    def load(self, path: str) -> 'FeatureExtractor':
        pass
