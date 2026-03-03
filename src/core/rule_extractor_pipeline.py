from typing import Callable, List, Dict, Any, Set, Optional, Union
from dataclasses import dataclass

import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier

from src.table.polyforest import PolyForest
from src.table.table_rule import TableRule
from src.table.compact.base_compact import BaseCompactOperator
from src.table.compact.simple_compact import GraduateBy, SimpleCompactOperator


RANDOM_SEED = 42


@dataclass
class LearnData:
    feature_cols: List[str]
    target_col: str 
    _train: pd.DataFrame
    _val: pd.DataFrame
    _eval: pd.DataFrame


@dataclass
class Rule:
    # _label: Callable[[...], str]
    _assign_label: Callable[[pd.DataFrame], pd.DataFrame]
    _labels: List[str]
    _result_col: str

    @classmethod
    def from_compact_op(cls, compact_op: BaseCompactOperator):
        assign_label = compact_op.assign_label
        labels = list(compact_op._compact_rule.values())
        result_col = compact_op.compact_label_field
        return cls(assign_label, labels, result_col)

    def assign_label(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._assign_label(df)

    @property
    def labels(self) -> List[str]:
        return self._labels
    
    @property
    def result_col(self) -> str:
        return self._result_col


class RuleExtractorPipeline:
    learn_data: LearnData = None
    cb_config: Dict[str, Any]
    project_path: str = ''
    cb_model: CatBoostRegressor | CatBoostClassifier = None
    polyforest: PolyForest = None
    table_rule: TableRule = None
    compact_op: BaseCompactOperator = None
    # _rule: Rule = None

    def __init__(self, learn_data: LearnData = None, project_path: str = '', cb_config: Dict[str, Any] = None):
        self.learn_data = learn_data
        self.project_path = project_path or self.project_path
        self.cb_config = cb_config or self._default_cb_config()

    def _path_to_project(self, relpath: Union[str, None]) -> str:
        return self.project_path + relpath if relpath else None

    def _default_cb_config(self) -> Dict[str, Any]:
        return {
            'iterations': 100,
            'learning_rate': 0.2,
            'depth': 6,
            'random_state': RANDOM_SEED,
            'loss_function': 'RMSE',
            'train_dir': self._path_to_project('/artifacts/learning/info/catboost_info'),
        }

    def _check_data(self, data: pd.DataFrame, feature_cols: List[str], target_col: str) -> None:
        if not set(feature_cols).issubset(data.columns):
            raise ValueError(f"Train data must contain features: {feature_cols}")
        if not target_col in data.columns:
            raise ValueError(f"Train data must contain target: {target_col}")
    
    def train(
        self,
        data: pd.DataFrame,
        feature_cols: List[str] = None,
        target_col: str = None,
        catboost_model = CatBoostRegressor,
        rel_save_path: str = '/artifacts/learning/info/catboost_model/active.cbm',
        **cb_params
    ) -> CatBoostRegressor | CatBoostClassifier:
        """
        Step 2.1: Train model to predict target.
        """
        save_path = self._path_to_project(rel_save_path)

        self.cb_model = catboost_model(**self.cb_config)

        feature_cols = feature_cols or self.learn_data.feature_cols
        target_col = target_col or self.learn_data.target_col
        self._check_data(data, feature_cols, target_col)

        X, y = data[feature_cols], data[target_col]
        self.cb_model.fit(X, y, **cb_params)

        if save_path is not None:
            self.cb_model.save_model(save_path)
        # TODO: # model.load_model('trained_model.cbm') 

        return self.cb_model

    def extract_polyforest(
        self,
        cb_model: CatBoostRegressor | CatBoostClassifier = None,
        truncate_at_k: int = None,
        rel_save_path: str = '/artifacts/learning/info/polynomial_form/active.yaml',
        rel_load_path: str = None
    ) -> PolyForest:
        """
        Step 2.2: Extract Polynomial Representation from (CatBoost) Decision Tree.
        """
        cb_model = cb_model or self.cb_model
        save_path = self._path_to_project(rel_save_path)
        load_path = self._path_to_project(rel_load_path)
        
        if load_path is not None:
            self.polyforest = PolyForest.load(load_path)
        elif cb_model is not None:
            if not cb_model.is_fitted():
                raise ValueError("Catboost model is not fitted.")
            self.polyforest = PolyForest.from_cb_model(cb_model)
        else:
            raise ValueError("PolyForest from ...?")

        if not truncate_at_k is None:
            self.polyforest.truncate(k=truncate_at_k)
        
        if not save_path is None:
            self.polyforest.save(save_path)
        
        return self.polyforest

    def extract_table_rule(
        self,
        polyforest: PolyForest = None,
        rel_load_path_to_polyforest: str = None
    ) -> TableRule:
        """
        Step 2.3: Extract Table Rule from Polynomial Representation of (CatBoost) Decision Tree.
        """
        load_path_to_polyforest = self._path_to_project(rel_load_path_to_polyforest)

        if load_path_to_polyforest is not None:
            polyforest = self.extract_polyforest(rel_load_path=rel_load_path_to_polyforest) 

        polyforest = polyforest or self.polyforest

        if polyforest is None:
            raise ValueError("From ... PolyForest?")

        self.table_rule = TableRule.from_polyforest(polyforest)
        return self.table_rule

    def extract_compact_op(
        self,
        compact_op_policy: str = 'SimpleCompactOperator',
        table_rule: TableRule = None,
        rel_save_path: str = '/artifacts/learning/info/compact/active.yaml',
        rel_load_path: str = None, 
        **op_params
    ) -> BaseCompactOperator:
        """
        Step 2.4: Compact Table Rule to constrained number of rules.
        """
        policies = {
            'SimpleCompactOperator': SimpleCompactOperator,
        }
        table_rule = table_rule or self.table_rule
        save_path = self._path_to_project(rel_save_path)
        load_path = self._path_to_project(rel_load_path)
        
        self.compact_op = policies[compact_op_policy](table_rule)
        
        if load_path is not None:
            self.compact_op.load_compact_rule(load_path)
        else:
            self.compact_op.fit(**op_params)

        if not save_path is None:
            self.compact_op.save_compact_rule(load_path or save_path)
            
        return self.compact_op

    def run(self) -> Rule:
        feature_cols = self.learn_data.feature_cols
        target_col = self.learn_data.target_col
        data_train, data_val, data_eval = (
            self.learn_data._train,
            self.learn_data._val,
            self.learn_data._eval,
        )

        grade =  GraduateBy(method='tail', threshold=0.85) # or `grade=GraduateBy(method='top_k', threshold=7)`
        op_kwargs = {
            'data': data_eval,
            'target_col': target_col,
            'grade':  grade,
            'with_rename': True,
        }
        
        # Step 2.1: Train model to predict target.
        self.train(
            data=data_train,
            eval_set=(data_val[feature_cols], data_val[target_col]),
            verbose=False,
            use_best_model=True
        )
        # Step 2.2: Extract Polynomial Representation from (CatBoost) Decision Tree.
        self.extract_polyforest(
            truncate_at_k=20,
        )
        # Step 2.3: Extract Table Rule from Polynomial Representation of (CatBoost) Decision Tree.
        self.extract_table_rule()
        # Step 2.4: Compact Table Rule to constrained number of rules.
        self.extract_compact_op(
            compact_op_policy='SimpleCompactOperator',
            **op_kwargs,
        )
        
        self._rule = Rule.from_compact_op(self.compact_op)
        return self._rule
    
    def load_rule(
        self,
        rel_path_to_polyforest: str = '/artifacts/learning/info/polynomial_form/active.yaml',
        rel_path_to_compact_op: str = '/artifacts/learning/info/compact/active.yaml',
    ) -> Rule:

        # Step 2.3: Extract Table Rule from Polynomial Representation of (CatBoost) Decision Tree.
        self.extract_table_rule(rel_load_path_to_polyforest=rel_path_to_polyforest)
        # Step 2.4: Compact Table Rule to constrained number of rules.
        self.extract_compact_op(rel_load_path=rel_path_to_compact_op)
        
        self._rule = Rule.from_compact_op(self.compact_op)
        return self._rule
    
    @property
    def rule(self):
        if not hasattr(self, '_rule'):
            raise ValueError("Rule has not been initialized yet.")
        return self._rule

    @rule.setter
    def rule(self, rule: Rule):
        if not isinstance(rule, Rule):
            raise ValueError("Rule must be an instance of Rule.")
        self._rule = rule