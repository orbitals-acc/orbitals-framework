from typing import List, Callable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes


class Statistics:
    """
    Get statistics from extracted data.
    """
    default_size_col: str = 'size'
    default_share_col: str = 'share'
    default_rank_col: str = 'rank'
    point_col: str
    _result: pd.DataFrame = None
    _compressed_result: pd.DataFrame = None

    def __init__(self, point_col: str = None):
        self.point_col = point_col

    def _get_aggs(
        self,
        size_col: str,
        target_col: str,
        extended: bool,
    ):
        def aggs(row) -> pd.Series:                                          
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
        return aggs
    
    def _calculate_by_point(
        self,
        data: pd.DataFrame,
        target_col: str, 
        group_col: str,
        extended: bool = True,
    ) -> pd.DataFrame:
        
        result = data.copy()
        result.fillna({target_col: 0}, inplace=True) # TODO: here..?

        size_col = self.default_size_col
        stat_aggs = self._get_aggs(size_col, target_col, extended=extended)
        result = (
            result
            .groupby(by=group_col)
            .apply(stat_aggs, include_groups=False)
        )

        share_col = self.default_share_col
        result[share_col] = result[size_col] / result[size_col].sum()
        result.sort_values(by=size_col, ascending=False, inplace=True)

        rank_col = self.default_rank_col
        result[rank_col] = result[share_col].rank(method='first', ascending=False)

        result.sort_values(target_col, ascending=False, inplace=True)
        return result
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:

        if self.point_col is None:
            self._result = self._calculate_by_point(data, *args, **kwargs)
            return self._result
        
        points = data.index.get_level_values(self.point_col).unique()
        result_points = []
        for point in points:
            result_point = self._calculate_by_point(data.loc[point], *args, **kwargs)
            result_point[self.point_col] = point
            result_points.append(result_point)

        self._result = pd.concat(result_points, axis=0)
        group_col = self._result.index.name
        self._result.reset_index(inplace=True)
        self._result.set_index([self.point_col, group_col], inplace=True)
        return self._result

    
    def summary(self) -> pd.DataFrame:
        if self._result is None:
            raise RuntimeError('Statistics is not calculated.')
        return self._result
    
    @staticmethod
    def solve(
        V: np.ndarray,
        D: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Decomposition Problem.
        """
        t, o = V.shape
        Y, W = np.log10(V), D
    
        z = np.median(Y, axis=0).reshape(1, -1)   # - orbital component  (1, O)
        s = np.zeros(t).reshape(-1, 1)            # - seasonal component (T, 1)
        w = np.mean(W, axis=0).reshape(1, -1)     # - weight component   (1, O)
        
        s = np.median(Y - z, axis=-1).reshape(-1, 1)
        s -= s.mean()
        z = np.median(Y - s, axis=0).reshape(1, -1)

        # hat_Y = z + s
        return 10 ** z, s, w
    
    def estimate(
        self,
        target_col: str, 
        groups: List[str],
    ) -> pd.DataFrame:
        
        if self.point_col is not None:
            summary_stats = self.summary().copy()
        else:
            raise RuntimeError('Estimates are based on points only.')
        
        group_idx = pd.IndexSlice[:, groups]
        share_col = self.default_share_col
        _, group_col = summary_stats.index.names
        
        V_matrix = summary_stats.loc[group_idx, target_col].values.reshape(len(groups), -1).T
        D_matrix = summary_stats.loc[group_idx, share_col].values.reshape(len(groups), -1).T

        v, s, w = self.solve(V_matrix, D_matrix)
        self._compressed_result = pd.DataFrame(data={group_col: groups, target_col: v.squeeze(), share_col: w.squeeze()})
        self._compressed_result.set_index(group_col, inplace=True)
        return self._compressed_result

    def compressed_summary(self,):
        if self._compressed_result is None:
            raise RuntimeError('Estimates is not computed yet.')
        return self._compressed_result
    
    def plot_bars(
        self, 
        height_col,
        width_col: str = None,
        point: str = None,
        compressed: bool = True,
        ascending: bool = True,
        log: bool = False
    ) -> Axes:
        
        width_col = width_col or self.default_share_col

        if self.point_col is None:
            summary_stats = self.summary().copy()
        elif point is not None:
            summary_stats = self.summary().loc[point].copy()
        elif compressed:
            summary_stats = self.compressed_summary().copy()
        else:
            raise ValueError('Statistics were not determined by point.')

        summary_stats.sort_values(height_col, ascending=ascending, inplace=True)

        fig, ax = plt.subplots(figsize=(12, 4))
        left_border = 0
        positions = []
        for label, row in summary_stats.iterrows():
            height, width = row[height_col], row[width_col]
            pos = left_border + width / 2
            ax.bar(pos, 
                height=height, width=width, label=label,
                alpha=0.7, edgecolor='black',
                log=log
            )
            left_border += width
            positions.append(pos)

        group_col = summary_stats.index.name
        ax.legend(title=group_col)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            summary_stats.index, 
            rotation=45,
            fontsize=11,
            fontweight='bold'
        )

        return ax
    
    def plot_series(
        self, 
        y_col: str,
        y_std_col: str = None,
        groups: list[str] = None,
        log: bool = False,
        show_if: Callable[[pd.Series], pd.Series] = None
    ) -> Axes:
    
        if self.point_col is not None:
            swap_summary_stats = self.summary().swaplevel(0, 1).copy()
        else:
            raise RuntimeError('Series are based on points only.')
        
        if show_if is not None:
            swap_summary_stats = show_if(swap_summary_stats)

        fig, ax = plt.subplots(figsize=(12, 4))
        group_col, _ = swap_summary_stats.index.names

        # TODO: order ...?
        if groups is None:
            groups = swap_summary_stats.index.get_level_values(group_col).unique()

        for group in groups:
            series = swap_summary_stats.loc[group][y_col]
            points, values = series.index, series.values
            line = ax.plot(points, values, '-o', label=group)[0]

            if y_std_col is not None:
                sigmas =  swap_summary_stats.loc[group][y_std_col].values
                sizes = swap_summary_stats.loc[group][self.default_size_col].values
                plt.fill_between(
                    line.get_xdata(),
                    values - 3 * sigmas / (sizes ** 0.5),
                    values + 3 * sigmas / (sizes ** 0.5),
                    alpha=0.1, color='blue'
            )

        ax.legend(title=group_col)
        if log:
            ax.set_yscale('log')
        return ax

