import pandas as pd
import numpy as np

from src.data.imputation import StrokeProfileImputer
from src.models.swimmer import Swimmer


class DataPipeline:
    def __init__(self, rules: dict):
        self.rules = rules
        self.imputer = StrokeProfileImputer()

    def load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        return self._clean(df)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        df = df.dropna(subset=["name"])
        return df

    def impute_missing_times(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.imputer.fit_transform(df)

    def build_roster(self, df: pd.DataFrame) -> list[Swimmer]:
        event_cols = [c for c in df.columns if c not in ("name", "age", "gender")]
        swimmers = []
        for _, row in df.iterrows():
            best_times = {col: float(row[col]) for col in event_cols if pd.notna(row[col])}
            swimmers.append(Swimmer(
                name=row["name"],
                age=int(row["age"]),
                gender=row["gender"],
                best_times=best_times,
            ))
        return swimmers

    def load_opponent_times(self, path: str) -> dict[tuple[str, str, str], list[float]]:
        """Load historical opponent times. Returns dict keyed by (age_group, gender, event)."""
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        projections: dict[tuple[str, str, str], list[float]] = {}
        for _, row in df.iterrows():
            key = (row["age_group"], row["gender"], row["event"])
            projections.setdefault(key, []).append(float(row["time_seconds"]))
        return projections
