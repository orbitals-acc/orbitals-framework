from typing import List, Any, Callable, Union, Dict

import pandas as pd

from src.table.polyforest import PolyForest


class TableRule:
    indicator_type: str
    indicators: List[str]
    indicate: Callable[[Union[pd.Series, Dict[str, Any]]], Union[pd.Series, Dict[str, Any]]]

    def __init__(self, 
                indicator_type: str,
                indicators:  List[str], 
                indicate: Callable[[Union[pd.Series, Dict[str, Any]]], Union[pd.Series, Dict[str, Any]]],
    ):
        self.indicator_type = indicator_type
        self.indicators = indicators
        self.indicate = indicate

    @classmethod
    def from_polyforest(cls, polyforest: PolyForest) -> 'TableRule':
        indicator_type, indicators, indicate = TableRule.build_indicator_from_polyforest(polyforest)
        return cls(indicator_type, indicators, indicate)
    
    @staticmethod
    def build_indicator_from_polyforest(polyforest: PolyForest) -> Callable[[Union[pd.Series, Dict[str, Any]]], str]:
        """Build an indicator function based on the polynomial form of the forest."""

        indicator_type = 'M'
        indicators = [f"{indicator_type}_{i}" for i in range(len(polyforest.monoms))]

        def indicate(row: Union[pd.Series, Dict[str, Any]]) -> Union[pd.Series, Dict[str, Any]]:
            """
            Indicate a row by each monomial in the polynomial form.
            """
            indicated_row = None
            code = polyforest.encode(row)
            if code == '0':
                indicated_row = {indicator: 0 for indicator in indicators}
            else:
                indicated_row = {indicator: int(value) for indicator, value in zip(indicators, code)}
            if isinstance(row, pd.Series):
                return pd.Series(indicated_row)
            return indicated_row
        
        return indicator_type, indicators, indicate
    
    def assign_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign indicators to extracted data.
        """
        df[self.indicators] = df.apply(self.indicate, axis=1)
        return df
    
    def dim(self):
        return len(self.indicators)
