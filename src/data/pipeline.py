import pandas as pd
import numpy as np

from src.data.imputation import StrokeProfileImputer
from src.models.swimmer import Swimmer


class DataPipeline:
    """Loads raw swimmer history, imputes missing times, normalizes to power points."""

    def __init__(self, rules: dict):
        self.rules = rules
        self.imputer = StrokeProfileImputer()

    def load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        return self.clean(df)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        df = df.dropna(subset=["name"])
        return df

    def impute_missing_times(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.imputer.fit_transform(df)

    def normalize_to_power_points(self, df: pd.DataFrame, base_times: dict[str, float]) -> pd.DataFrame:
        """Convert raw times to power points relative to a base standard time."""
        normed = df.copy()
        for event, base in base_times.items():
            if event in normed.columns:
                normed[event] = (base / normed[event]) * 1000
        return normed

    def build_roster(self, df: pd.DataFrame) -> list[Swimmer]:
        swimmers = []
        for _, row in df.iterrows():
            times = {col: row[col] for col in df.columns if col != "name" and pd.notna(row[col])}
            swimmers.append(Swimmer(name=row["name"], best_times=times))
        return swimmers
