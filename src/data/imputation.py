import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class StrokeProfileImputer:
    """
    Fills missing event times using linear regression against a known event in the same distance group.
    Only imputes within the correct distance tier (25-yard vs 50-yard) so young swimmers
    never get 50-yard times predicted and vice versa.
    """

    PREDICTOR_MAP: dict[str, list[str]] = {
        # 25-yard events (6U, 7-8, 9-10)
        "25_butterfly":    ["25_freestyle"],
        "25_backstroke":   ["25_freestyle"],
        "25_breaststroke": ["25_freestyle"],
        # 50-yard events (11-12, 13-14, 15-18)
        "50_butterfly":    ["50_freestyle"],
        "50_backstroke":   ["50_freestyle"],
        "50_breaststroke": ["50_freestyle"],
        "100_im":          ["50_freestyle"],
    }

    def __init__(self):
        self._models: dict[str, tuple] = {}

    def fit(self, df: pd.DataFrame) -> "StrokeProfileImputer":
        for target, predictors in self.PREDICTOR_MAP.items():
            available = [p for p in predictors if p in df.columns]
            if target not in df.columns or not available:
                continue
            train = df[[target] + available].dropna()
            if len(train) < 3:
                continue
            model = LinearRegression()
            model.fit(train[available], train[target])
            self._models[target] = (model, available)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for target, (model, predictors) in self._models.items():
            if target not in df.columns:
                df[target] = np.nan
            missing_mask = df[target].isna()
            has_predictors = df[predictors].notna().all(axis=1)
            to_impute = missing_mask & has_predictors
            if to_impute.any():
                df.loc[to_impute, target] = model.predict(df.loc[to_impute, predictors])
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
