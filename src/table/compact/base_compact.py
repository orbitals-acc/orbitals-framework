from abc import ABC, abstractmethod
from typing import Dict, Union, Any, Callable

import yaml
import pandas as pd
from collections import OrderedDict

from src.table.table_rule import TableRule


class BaseCompactOperator(ABC):
    table_rule: TableRule
    _compact_rule: OrderedDict[str, str] = None
    table_label_field: str = 'table_label'
    compact_label_field: str = 'orbit'

    def __init__(self, table_rule: TableRule):
        self.table_rule = table_rule

    def assign_table_label(self, df: pd.DataFrame) -> pd.DataFrame:
        if not all(item in df.columns for item in self.table_rule.indicators):
            _df = self.table_rule.assign_indicators(df.copy())
            return df.assign(**{self.table_label_field: _df[self.table_rule.indicators].astype(str).agg(''.join, axis=1)})
        return df.assign(**{self.table_label_field: df[self.table_rule.indicators].astype(str).agg(''.join, axis=1)})
    
    @abstractmethod
    def label(self, table_label: str) -> str:
        pass

    def assign_label(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.table_label_field in df.columns:
            _df = self.assign_table_label(df.copy())
            return df.assign(**{self.compact_label_field: _df[self.table_label_field].apply(self.label)})
        return df.assign(**{self.compact_label_field: df[self.table_label_field].apply(self.label)})

    @abstractmethod
    def fit(data: pd.DataFrame) -> OrderedDict[str, str]:
        pass

    @property
    def compact_rule(self) -> OrderedDict[str, str]:
        return self._compact_rule
    
    def save_compact_rule(self, filepath: str) -> None:
        """
        Save the compact rule to a YAML file.
        """
        with open(filepath, 'w') as f:
            yaml.dump(dict(self._compact_rule), f, default_flow_style=False)

    def load_compact_rule(self, filepath: str) -> OrderedDict[str, str]:
        """
        Load the compact rule from a YAML file.
        """
        with open(filepath, 'r') as f:
            self._compact_rule = OrderedDict(yaml.safe_load(f))
        return self._compact_rule
