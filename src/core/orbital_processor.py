from typing import Dict, Any, Union, List
from collections.abc import Hashable

import time
import yaml
import pandas as pd
import datetime as datetime

from src.core.feature_extractor import FeatureDataStore, FeatureExtractor
from src.core.rule_extractor_pipeline import Rule
from src.evaluation.statistics import Statistics


class OrbitalProcessor:
    feature_extractor: FeatureExtractor
    rule: Rule
    _labeled_data: pd.DataFrame = None
    # _target_col: str = None
    _hat_potentials: pd.Series = None

    def __init__(
            self, 
            feature_extractor: FeatureExtractor, 
            rule: Rule,
            dirname: str,
            project_path: str,
            stats: Statistics = None,
        ) -> None:
        self.feature_extractor = feature_extractor
        self.rule = rule
        self.dirname = dirname
        self.project_path = project_path
        self.stats = stats or Statistics('timestamp')

    def evaluate(
        self, 
        timestamps: List[Union[datetime.datetime, str]],
        users: List[Hashable] = None,
        with_target: bool = True, 
        verbose: bool = False,
        freeze: bool = True
    ) -> pd.DataFrame:
        
        self.feature_extractor.clear()
        start_time = time.time()

        # 1. Extract Data
        extracted_data = self.feature_extractor.extract_data(
            timestamps=timestamps,        # (..., ts] & (ts, ...]
            users=users,                  
            with_target=with_target 
        )

        if verbose:
            print(f'✔ : Extract Data: {time.time() - start_time:.3f}s')
        start_time = time.time()

        # 2. Label Extracted Data
        labeled_data = self.rule.assign_label(extracted_data)

        if verbose:
            print(f'✔ : Label Data: {time.time() - start_time:.3f}s')
        start_time = time.time()

        if freeze:
            self._labeled_data = labeled_data

        return labeled_data
    
    def _path_to_project(self, relpath: Union[str, None]) -> str:
        return self.project_path + relpath if relpath else None
    
    def _relpath(self, folder: str, dirname: str) -> str:
        return f'/artifacts/evaluation/stream/{folder}/{dirname}'

    def save_labeled_data(self)-> None:

        if self._labeled_data is None:
            raise ValueError("Labeled data has not been initialized yet.")
        
        relpath = self._relpath('data', self.dirname)
        dirpath = self._path_to_project(relpath)

        timestamps = self._labeled_data.index.get_level_values('timestamp').unique()
        for ts in timestamps:
            filepath = f"{dirpath}/{str(ts)}.csv"
            self._labeled_data.loc[ts].to_csv(filepath)
        print(f'Save to {dirpath}')
        return

    def load_labeled_data(self) -> pd.DataFrame:
        relpath = self._relpath('data', self.dirname)
        dirpath = self._path_to_project(relpath)

        import glob
        import os
        filepattern = f'{dirpath}/*.csv'
        table_paths = glob.glob(filepattern)

        # timestamp_col = self.feature_extractor.session_processor.timestamp_col
        timestamp_col = 'timestamp' # TODO: support origin col
        user_id_col = self.feature_extractor.session_processor.user_id_col

        tables = []
        for table_path in table_paths:
            timestamp = os.path.splitext(os.path.basename(table_path))[0]
            df = pd.read_csv(table_path)
            df[timestamp_col] = timestamp
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            tables.append(df)

        labeled_data = pd.concat(tables, ignore_index=True)
        labeled_data.sort_values(by=[timestamp_col], inplace=True)
        labeled_data.set_index([timestamp_col, user_id_col], inplace=True)

        self._labeled_data = labeled_data
        print(f'Load from {dirpath}')
        return self._labeled_data

    @property
    def labeled_data(self) -> pd.DataFrame:
        if self._labeled_data is None:
            raise RuntimeError('Labeled data not loaded yet.')
        return self._labeled_data

    def estimate(
        self,
        target_col: str,     
    ) -> pd.Series:
        self.stats.calculate(self.labeled_data, target_col, group_col=self.rule.result_col)
        self.stats.estimate(target_col, groups=self.rule.labels)
        # self.stats.summary()
        self._hat_potentials = self.stats.compressed_summary()[target_col]
        return self._hat_potentials
    
    @property
    def hat_potentials(self) -> pd.Series:
        if self._hat_potentials is None:
            raise RuntimeError('Potentials is not estimated yet.')
        return self._hat_potentials
    
    def hats_to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.hat_potentials.name,
            'values': self.hat_potentials.to_dict(),
            'index': list(self.hat_potentials.index),
            'index_name': self.hat_potentials.index.name,
        }

    def hats_from_dict(self, data: Dict[str, Any]) -> pd.Series:
        hats = pd.Series(data=data['values'], index=data['index'])
        hats.index.name = data['index_name']
        hats.name = data['name']
        return hats

    def save_hats(self) -> None:
        relpath = self._relpath('potentials', self.dirname)
        dirpath = self._path_to_project(relpath)
        filepath = f"{dirpath}/active.yaml"
        data = self.hats_to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        return

    def load_hats(self) -> pd.Series:
        relpath = self._relpath('potentials', self.dirname)
        dirpath = self._path_to_project(relpath)
        filepath = f"{dirpath}/active.yaml"
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        self._hat_potentials = self.hats_from_dict(data)
        return self._hat_potentials
    
    def assign_hat_potentials(self, df: pd.DataFrame) -> pd.DataFrame:
        orbit_col = self.hat_potentials.index.name
        hat_col = f"hat_{self.hat_potentials.name}"
        if orbit_col not in df.columns:
            raise ValueError('Input df is not labelled.') 
        return df.assign(**{hat_col: df[orbit_col].map(self.hat_potentials.to_dict())})