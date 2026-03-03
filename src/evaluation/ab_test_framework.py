from types import NoneType
from typing import Dict, Any, List, Tuple
from collections.abc import Hashable

import pandas as pd
import numpy as np
from datetime import datetime
import scipy.stats as ss

from src.core.orbital_processor import OrbitalProcessor


class ABTestFramework:
    orbital_processor: OrbitalProcessor
    target: str
    default_metrics: List[Tuple[str, str]]
    groups_data : pd.DataFrame = None
    _report: pd.DataFrame = None

    def __init__(self, orbital_processor: OrbitalProcessor) -> None:
        self.orbital_processor = orbital_processor
        self.target = self._get_target()
        self.default_metrics = self._default_metrics()

    def _get_target(self):
        return self.orbital_processor.hat_potentials.name
    
    def _default_metrics(self):
        return [
        (f'hat_{self.target}_uplift', f'hat_{self.target}_at_T'),
        # (f'{self.target}_at_T', None)
    ]

    def _merge(self, data_at_T: pd.DataFrame, data_at_0: pd.DataFrame,) -> pd.DataFrame:
        data_at_0.columns = [f"{col}_at_0" for col in data_at_0]
        data_at_T.columns = [f"{col}_at_T" for col in data_at_T]
        merged_data = pd.concat([data_at_0, data_at_T], axis=1)
        hat_col = f"hat_{self.target}"
        merged_data[f"{hat_col}_uplift"] = merged_data[f"{hat_col}_at_T"] - merged_data[f"{hat_col}_at_0"]
        return merged_data

    def calculate_group(
        self,
        start: datetime | str,
        end: datetime | str,
        users: List[Hashable],
        with_target: bool = False
    ) -> pd.DataFrame:
        data = self.orbital_processor.evaluate(
            timestamps=[start, end],
            users=users,
            with_target=with_target,
            verbose=False,
            freeze=False,
        )
        self.orbital_processor.feature_extractor.clear() # (optional)
        data = self.orbital_processor.assign_hat_potentials(data)
        data = self._merge(data.loc[end], data.loc[start])
        return data
        
    def calculate(
        self,
        start: datetime | str,
        end: datetime | str,
        groups: Dict[Hashable,  List[Hashable]],
        # control: Hashable, 
        with_target: bool = False
    ) -> Dict[str, Any]:
        groups_data = {}

        for variant, users in groups.items():
            groups_data[variant] = self.calculate_group(start, end, users, with_target=with_target)

        self.groups_data = groups_data
        return self.groups_data
    

    def get_report(
        self,
        variants: List[str],
        control: str,
        metrics: List[Tuple[str, str]] = None,
        proportional: bool = False,
        **kwargs
    ) -> None:
        metrics = metrics or self.default_metrics
        reports = []
        for variant in variants:
            if variant == control:
                continue 
            for rel_value_col, abs_value_col in metrics:
                report = self.get_bootstrap_significance(
                    rel_value_col = rel_value_col,
                    data = self.groups_data,
                    test = variant,
                    control = control,
                    proportional = proportional,
                    abs_value_col = abs_value_col,
                    **kwargs
                )
                reports.append(report)
        self._report = pd.DataFrame(reports)
        return self._report

    @property
    def report(self):
       if self._report is None:
           raise RuntimeError('Report is not calculated yet. Please call the `report` method first.')
       return self._report
    
    @staticmethod
    def get_boostrap_means(
        value_cols: list[str],
        data: pd.DataFrame,
        users: np.ndarray = None,
        n_resamples: int = 3_000,
        seed: int = 42
    ) -> np.ndarray:
        
        np.random.seed(seed)
        
        for value_col in value_cols:
            data[value_col] = data[value_col].fillna(0)

        if users is None:
            users = np.asarray(data.index)
            
        resample_idx = np.random.choice(users, size=n_resamples * users.size)
        b_values = data[value_cols].loc[resample_idx].values.reshape(n_resamples, -1, len(value_cols))
        b_means = b_values.mean(axis=1)
        return b_means.T

    @staticmethod
    def get_bootstrap_significance(
        rel_value_col: str,
        data: pd.DataFrame,
        test: str,
        control: str = 'control',
        proportional: bool = False,
        abs_value_col: str = None,
        alpha = 0.05, gamma=0.05,
        **kwargs,
    ):
        abs_value_col = abs_value_col or rel_value_col

        mean_x_B = np.mean(data[test][rel_value_col].fillna(0))
        mean_x_A = np.mean(data[control][rel_value_col].fillna(0))
        abs_mean_x_A = np.mean(data[control][abs_value_col].fillna(0))
        
        sample_B = ABTestFramework.get_boostrap_means([rel_value_col], data[test], **kwargs)
        sample_A, abs_sample_A = ABTestFramework.get_boostrap_means([rel_value_col, abs_value_col], data[control], **kwargs)

        if proportional:        
            sample = 100 * (sample_B - sample_A) / abs_sample_A
            delta = 100 * (mean_x_B - mean_x_A) / abs_mean_x_A
        else:
            sample = sample_B - sample_A
            delta = mean_x_B - mean_x_A

        ci_l, ci_r = np.quantile(sample, alpha / 2), np.quantile(sample, 1 - alpha / 2)
        x_l, x_r = np.quantile(sample, gamma), np.quantile(sample, 1 - gamma)
        sigma = (x_r - x_l) / (ss.norm.ppf(1 - gamma) - ss.norm.ppf(gamma))

        alpha_0 = (0 <= sample).mean()
        pv = 2 * min(alpha_0, 1 - alpha_0)
        
        return {
            "control": control,
            "test": test,
            "metric": rel_value_col,
            "∆": delta,
            "p-value": pv,
            "σ": sigma,
            "ci": [ci_l, ci_r],
            "sample": sample,
            "kind": f'%' if proportional else '',
        }

