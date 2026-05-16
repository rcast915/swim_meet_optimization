import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class StrokeProfileImputer:
    """
    Extrapolates missing event times using linear regression against known events.

    Example: predict 200 IM from 100 Breaststroke + 100 Butterfly splits.
    Each target event gets its own regression model fit on athletes who have
    all required values.
    """

    # Maps target event -> predictor events used for regression
    PREDICTOR_MAP: dict[str, list[str]] = {
        "200_im": ["100_butterfly", "100_breaststroke"],
        "500_freestyle": ["200_freestyle", "100_freestyle"],
        "200_freestyle": ["100_freestyle", "50_freestyle"],
        "100_butterfly": ["50_freestyle", "100_freestyle"],
        "100_backstroke": ["50_freestyle", "100_freestyle"],
        "100_breaststroke": ["50_freestyle", "100_freestyle"],
    }

    def __init__(self):
        self._models: dict[str, LinearRegression] = {}

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
