from dataclasses import dataclass, asdict, field
from typing import Union, Any, List, Set, Dict, Optional

import re
import yaml
import pandas as pd

from catboost import CatBoostRegressor, monoforest
from catboost._catboost import Split, Monom


class PolyForest:
    # model: CatBoostRegressor = None
    monoms: List[Monom]
    feature_names: List[str]
    is_truncated: bool = False

    def __init__(self, monoms: List[Monom], feature_names: List[str], size: float = None):
        self.monoms = monoms
        self.feature_names = feature_names
        self.size = size or max(m.weight for m in monoms)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "size" : self.size, 
            "feature_names": self.feature_names,
            "monoms": [
                {
                    "splits": [
                        {
                            'feature_idx': split.feature_idx,
                            'split_type': split.split_type,
                            'border': split.border,
                        }
                         for split in monom.splits],
                    "value": monom.value,
                    "weight": monom.weight
                }
                for monom in self.monoms
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PolyForest':
        monoms = []
        for item in data["monoms"]:
            splits = [Split(**split) for split in item["splits"]]
            monom = Monom(
                splits=splits,
                value=item["value"],
                weight=item["weight"]
            )
            monoms.append(monom)
        return cls(monoms=monoms, feature_names=data["feature_names"], size=data.get('size', None))

    def save(self, filepath: str) -> None:
        """
        Save as YAML file.
        """
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load(cls, filepath: str) -> 'PolyForest':
        """
        Load from YAML file.
        """
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_cb_model(cls, cb_model: CatBoostRegressor) -> 'PolyForest':
        if not cb_model.is_fitted():
            raise RuntimeError("Catboost model is not fitted.")
        return cls(monoms=monoforest.to_polynom(cb_model), feature_names=cb_model.feature_names_)
    
    def get_feature_name(self, idx: int) -> str:
        if idx < 0 or idx >= len(self.feature_names):
            return f'F{idx}'
        return self.feature_names[idx]
    
    def is_satisfy(self, row: Union[pd.Series, Dict[str, Any]], m: Monom) -> Optional[bool]:
        """
        Check if the given row satisfies the monomial.
        """
        flag = True
        for split in m.splits:
            idx = split.feature_idx
            feature_name = self.get_feature_name(idx)
            if pd.isna(row[feature_name]): # TODO: is None! => out
                return None
            flag &= (row[feature_name] > split.border)
        return flag
    
    def encode(self, row: Union[pd.Series, Dict[str, Any]]) -> str:
        """
        Encode a row into a string representation of monomials.
        """
        code = ''
        for m in self.monoms:
            flag = self.is_satisfy(row, m)
            if flag is None:
                return '0'
            code += str(int(flag))
        return code

    def truncate(self, k = None) -> None:
        """
        Truncate the polynomial to the top k monoms based on their importance scores.
        """
        if k is not None:
            key = lambda m: -m.weight * (m.value[0] ** 2)
            self.monoms = sorted(self.monoms, key = key)[:k]
            self.is_truncated = True
        return
    
    def _read_monomial_repr(self, m: Monom):
        str_m = ''.join(map(lambda s: str(s), m.splits)) 
        for idx in range(len(self.feature_names)):
            feature_name = self.get_feature_name(idx)
            str_m = re.sub(f'F{idx}', feature_name, str_m)
        return str_m
    
    def __repr__(self):
        res = ''
        key = lambda m: -m.weight * (m.value[0] ** 2)
        for m in sorted(self.monoms, key=key):
            max_weight = self.size
            p_m = m.weight / max_weight
            str_m = self._read_monomial_repr(m)
            res += f"p(m):{100 * (p_m):.2f}% | E(w^2):{p_m * m.value[0] ** 2:.6f} | ({m.value[0]}) * {str_m} \n"
        return res
    
    def summary(self):
        from tabulate import tabulate
        
        if not self.monoms:
            return "Empty model"
        
        # Prepare data
        key = lambda m: -m.weight * (m.value[0] ** 2)
        sorted_monoms = sorted(self.monoms, key=key)
        max_weight = self.size
        
        # Create table
        rows = []
        for idx, m in enumerate(sorted_monoms):
            p_m = m.weight / max_weight
            str_m = self._read_monomial_repr(m)
            conditions = str_m if str_m else "Intercept"
            
            rows.append([
                f"{idx}",
                f"{m.value[0]:.10f}",
                f"{100 * p_m:.2f}",
                f"{p_m * m.value[0] ** 2:.6f}",
                conditions
            ])
        
        # Header
        summary = "\n" + "="*150 + "\n"
        summary += "POLYNOMIAL REPRESENTATION SUMMARY".center(150) + "\n"
        summary += "="*150 + "\n\n"
        
        # Table
        headers = ["Index", "Coefficient", "p(M), %", "E(w²M)", "Conditions"]
        summary += tabulate(rows, headers=headers, tablefmt="grid", disable_numparse=True)
        summary += "\n" + "="*150 + "\n"
        
        # Statistics
        total_weight = sum(m.weight for m in self.monoms)
        num_terms = len(self.monoms)
        coeffs = [m.value[0] for m in self.monoms]
        mean_coef = sum(coeffs) / num_terms
        std_coef = (sum((c - mean_coef)**2 for c in coeffs) / num_terms) ** 0.5
        
        summary += "\nSTATISTICS\n"
        summary += "-"*150 + "\n"
        summary += f"{'Number of terms:':<40} {num_terms:>15}\n"
        summary += "\n"
        
        return summary

