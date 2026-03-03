from abc import ABC, abstractmethod
from typing import Dict, Union, Any, Callable, Tuple

import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes

from src.table.compact.base_compact import BaseCompactOperator


class GraduateBy:
    method: str
    threshold: float
    field_name: str = None

    def __init__(self, method: str, threshold: float, field_name: str = None):
        self.method = method
        self.threshold = threshold
        self.field_name = field_name

    @staticmethod
    def truncate(
        stats: pd.DataFrame,
        weight_col: str,
        threshold: float,
        density: bool = False,
        cumulative: bool = False,
        sort_col: str = None,
        ascending: bool = True
    ) -> pd.DataFrame: 
        """
        Examples:
        (1) Truncate last (n - k)
        `custom(stats, 'rank', k)`
        (2) Truncate (1 - q) tail
        `custom(stats, 'size', q, density=True, cumulative=True, ascending=False)`
        """
        sort_col = sort_col or weight_col
        stats = stats.sort_values(sort_col, ascending=ascending)
        weights = stats[weight_col].values.copy()
        if density:
            weights /= weights.sum()
        if cumulative: 
            weights = weights.cumsum()
        return stats.iloc[weights <= threshold]

    def do_truncate(self, stats: pd.DataFrame) -> pd.DataFrame:
        if self.method == 'top_k':
            field_name = 'rank' or self.field_name
            return GraduateBy.truncate(stats, field_name, self.threshold)
        elif self.method == 'tail':
            field_name = 'size' or self.field_name
            return GraduateBy.truncate(stats, field_name, self.threshold, density=True, cumulative=True, ascending=False)
        else:
            ValueError('Truncate method not implemented')

    @staticmethod
    def rename(
        stats: pd.DataFrame,
        weight_col: str,
        thresholds: OrderedDict[str, float],
        density: bool = False,
        cumulative: bool = False,
        sort_col: str = None,
        ascending: bool = True,
    ) -> OrderedDict[str, str]:
        """
        Examples:
        (1) ...
        (2) ...
        """
        sort_col = sort_col or weight_col
        stats = stats.sort_values(sort_col, ascending=ascending)
        weights = stats[weight_col].values.copy()

        if density:
            weights /= weights.sum()
        if cumulative: 
            weights = weights.cumsum()

        level_col = 'level'
        bins = list(thresholds.values()) + [weights[-1] + 1]
        labels = list(thresholds.keys())
        stats[level_col] = pd.cut(weights, bins=bins, labels=labels)

        sublevel_col = 'sublevel'
        stats[sublevel_col] = stats.groupby(level_col, observed=True).cumcount() + 1
        stats[sublevel_col] = stats[sublevel_col].astype(str)

        old_index = stats.index
        new_index = stats.apply(lambda row: row[level_col] + row[sublevel_col], axis=1)
        # rename_dict = {table_label:new_label for table_label, new_label in zip(old_index, new_index)}
        rename_dict = OrderedDict(list(zip(old_index, new_index)))
        return rename_dict
    
    @property
    def default_thresholds(self) -> OrderedDict[str, float]:
        marks = ['A', 'B', 'C', 'D', 'E', 'F']
        borders = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        return OrderedDict(list(zip(marks, borders)))

    def do_rename(self, stats: pd.DataFrame, sort_col: str) -> Dict[str, str]: 
        return GraduateBy.rename(stats, 'size', self.default_thresholds, density=True, cumulative=True, sort_col=sort_col, ascending=False)


class SimpleCompactOperator(BaseCompactOperator):
    default_out_label: str = 'out'
    default_other_label: str = 'other'
    _summary: pd.DataFrame = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def label(self, table_label: str) -> str:
        if table_label == '0' * self.table_rule.dim(): # TODO: is None! => out
            return self.default_out_label
        return self._compact_rule.get(table_label, self.default_other_label)
    
    def _full_compact_rule(self) -> None:
        self._compact_rule['0' * self.table_rule.dim()] = self.default_out_label
        self._compact_rule[self.default_other_label] = self.default_other_label
        return
    
    # TODO: remove code duplicates!
    def get_stats_aggs(
            self,
            size_col: str,
            target_col: str,
            extended: bool,
        ):

        def stats_aggs(row) -> pd.Series:                                          
            stat_funcs = {
                **{
                size_col: row[target_col].count(),
                target_col : row[target_col].mean(),
                },
                **(
                    {
                        f'{target_col}_std': row[target_col].std(),
                        # f'{target_col}_var': row[target_col].var(),
                        f'{target_col}_>0': (row[target_col] > 0).mean(),
                    }
                    if extended else {}
                )
            }
            return pd.Series(stat_funcs)

        return stats_aggs

    def get_stats(
            self,
            data: pd.DataFrame,
            target_col: str, 
            group_col: str = None,
            extended: bool = True,
    ) -> pd.DataFrame:
        
        stats = data.copy()

        if group_col is not None:
            group_col = group_col
        elif self.compact_label_field in data.columns:
            group_col = self.compact_label_field
        elif self.table_label_field in data.columns:
            group_col = self.table_label_field
        else:
            ValueError('Group by ...?')

        size_col='size'
        stat_aggs = self.get_stats_aggs(size_col, target_col, extended=extended)
        stats = (
            stats
            .groupby(by=group_col)
            .apply(stat_aggs, include_groups=False)
        )

        share_col='share'
        stats[share_col] = stats[size_col] / stats[size_col].sum()
        stats = stats.sort_values(by=size_col, ascending=False)

        rank_col = 'rank'
        stats[rank_col] = stats[share_col].rank(method='first', ascending=False)

        return stats
    
    def fit(self, 
            data: pd.DataFrame,
            target_col: str,
            grade: GraduateBy,
            with_rename: bool = False,
            with_summary: bool = True,
        ) -> OrderedDict[str, str]:
        if not self.table_label_field in data.columns:
            data = self.assign_table_label(data)
        stats = self.get_stats(data, target_col, group_col=self.table_label_field, extended=True)
        truncated_stats = grade.do_truncate(stats)
        if with_rename:
            self._compact_rule = grade.do_rename(truncated_stats, sort_col=target_col)
        else:
            # self._compact_rule = {table_label:table_label for table_label in zip(truncated_stats.index)}
            self._compact_rule = OrderedDict(list(zip(truncated_stats.index, truncated_stats.index)))
        self._full_compact_rule()
        if with_summary:
            self.get_summary_stats(data, target_col)
        return self._compact_rule
    
    def get_summary_stats(
            self, 
            data: pd.DataFrame, 
            target_col: str,
            rewrite: bool = True
    ) -> pd.DataFrame:
        if self._compact_rule is None:
            RuntimeError('Compact operator is not fitted.')
        summary_stats = self.get_stats(self.assign_label(data), target_col, group_col=self.compact_label_field, extended=True)
        summary_stats = summary_stats.sort_values(target_col, ascending=False)
        if rewrite:
            self._summary = summary_stats
        return summary_stats

    @property
    def summary(self):
        if self._summary is not None:
            return self._summary
        raise RuntimeError('Summary is not calculated.')

    def plot(self, height_col: str, width_col: str, log: bool = False) -> Axes:
        fig, ax = plt.subplots(figsize=(12, 4))

        summary_stats = self.summary.sort_values(height_col, ascending=True)
        left_border = 0
        positions = []
        for label, row in summary_stats.iterrows():
            height, width = row[height_col], row[width_col]
            pos = left_border + width/2
            ax.bar(pos, height=height, width=width, label=label,
                alpha=0.7, edgecolor='black',
                log=log
            )
            left_border += width
            positions.append(pos)

        ax.legend(title=summary_stats.index.name)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            summary_stats.index, 
            rotation=45,
            fontsize=11,
            fontweight='bold'
        )

        return ax
