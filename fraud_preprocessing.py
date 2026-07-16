"""Minimal workpackage-1 preprocessing for IEEE-CIS fraud data.

This file intentionally keeps the data-prep path small: load and join the
competition CSVs, normalize known naming quirks, and convert every feature
column into a numeric matrix with a fitted, reusable encoder.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd


TRANSACTION_CATEGORICAL_COLUMNS = [
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
]
IDENTITY_CATEGORICAL_COLUMNS = (
    ["DeviceType", "DeviceInfo"] + [f"id_{i:02d}" for i in range(12, 39)]
)
BOOLEAN_LIKE_VALUES = {"T": 1, "F": 0, True: 1, False: 0}
MISSING_TOKEN = "__missing__"


class FraudDataLoader:
    """Load the IEEE-CIS split CSVs and return one joined training table."""

    required_train_files = ("train_transaction.csv", "train_identity.csv")
    required_test_files = ("test_transaction.csv", "test_identity.csv")

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)

    def load_train(self, nrows: int | None = None) -> pd.DataFrame:
        self._require_files(self.required_train_files)
        transaction = pd.read_csv(self.data_dir / "train_transaction.csv", nrows=nrows)
        identity = pd.read_csv(self.data_dir / "train_identity.csv", nrows=nrows)
        return self._join_transaction_identity(transaction, identity, require_target=True)

    def load_test(self, nrows: int | None = None) -> pd.DataFrame:
        self._require_files(self.required_test_files)
        transaction = pd.read_csv(self.data_dir / "test_transaction.csv", nrows=nrows)
        identity = pd.read_csv(self.data_dir / "test_identity.csv", nrows=nrows)
        return self._join_transaction_identity(transaction, identity, require_target=False)

    def load_fraud_only(self, nrows: int | None = None) -> pd.DataFrame:
        return self.load_train(nrows=nrows).query("isFraud == 1").copy()

    def _join_transaction_identity(
        self, transaction: pd.DataFrame, identity: pd.DataFrame, require_target: bool
    ) -> pd.DataFrame:
        identity = self.normalize_identity_columns(identity)
        required_transaction_columns = ["TransactionID"]
        if require_target:
            required_transaction_columns.append("isFraud")
        self._require_columns(transaction, required_transaction_columns, "transaction data")
        self._require_columns(identity, ["TransactionID"], "identity data")

        joined = transaction.merge(
            identity, on="TransactionID", how="left", indicator="_identity_join"
        )
        joined["has_identity"] = (joined["_identity_join"] == "both").astype(int)
        joined = joined.drop(columns=["_identity_join"])
        return self._cast_known_categoricals(joined)

    @staticmethod
    def normalize_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            column: column.replace("id-", "id_", 1)
            for column in df.columns
            if column.startswith("id-")
        }
        return df.rename(columns=rename_map)

    def _require_files(self, filenames: tuple[str, ...]) -> None:
        missing = [name for name in filenames if not (self.data_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing required data files: {missing}")

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: list[str], filename: str) -> None:
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(f"{filename} is missing required columns: {missing}")

    @staticmethod
    def _cast_known_categoricals(df: pd.DataFrame) -> pd.DataFrame:
        categorical_columns = set(TRANSACTION_CATEGORICAL_COLUMNS) | set(
            IDENTITY_CATEGORICAL_COLUMNS
        )
        present = [column for column in categorical_columns if column in df.columns]
        out = df.copy()
        out[present] = out[present].astype("category")
        return out


class NumericPreprocessor:
    """Fit once, then transform train/validation/test data to the same numeric columns."""

    def __init__(
        self,
        exclude_columns: list[str] | None = None,
        max_onehot_cardinality: int = 30,
    ) -> None:
        self.exclude_columns = exclude_columns or ["TransactionID", "isFraud"]
        self.max_onehot_cardinality = max_onehot_cardinality
        self.column_plan_: dict[str, dict] = {}
        self.output_columns_: list[str] = []

    def fit(self, df: pd.DataFrame) -> "NumericPreprocessor":
        self.column_plan_ = {}
        for column in df.columns:
            if column in self.exclude_columns:
                continue
            series = df[column]
            if self._is_boolean_like(series):
                self.column_plan_[column] = {"kind": "boolean"}
            elif pd.api.types.is_numeric_dtype(series):
                median = series.median()
                self.column_plan_[column] = {
                    "kind": "numeric",
                    "median": float(median) if pd.notna(median) else 0.0,
                }
            else:
                filled = self._as_category_strings(series)
                categories = sorted(filled.unique().tolist())
                if len(categories) <= self.max_onehot_cardinality:
                    self.column_plan_[column] = {
                        "kind": "onehot",
                        "categories": categories,
                    }
                else:
                    frequencies = filled.value_counts(normalize=True).to_dict()
                    self.column_plan_[column] = {
                        "kind": "frequency",
                        "frequencies": frequencies,
                    }
        self.output_columns_ = list(self.transform(df).columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        encoded: dict[str, pd.Series] = {}
        for column, plan in self.column_plan_.items():
            if column not in df.columns:
                raise ValueError(f"Input data is missing fitted column: {column}")
            series = df[column]
            if series.isna().any():
                encoded[f"{column}__was_missing"] = series.isna().astype(int)
            elif f"{column}__was_missing" in self.output_columns_:
                encoded[f"{column}__was_missing"] = pd.Series(0, index=df.index)

            if plan["kind"] == "boolean":
                values = series.astype("object").map(BOOLEAN_LIKE_VALUES)
                encoded[column] = values.fillna(-1).astype(int)
            elif plan["kind"] == "numeric":
                encoded[column] = pd.to_numeric(series, errors="coerce").fillna(plan["median"])
            elif plan["kind"] == "onehot":
                values = self._as_category_strings(series)
                for category in plan["categories"]:
                    encoded[f"{column}__{category}"] = (values == category).astype(int)
            elif plan["kind"] == "frequency":
                values = self._as_category_strings(series)
                encoded[f"{column}__frequency"] = (
                    values.map(plan["frequencies"]).fillna(0.0).astype(float)
                )

        out = pd.DataFrame(encoded, index=df.index)
        if self.output_columns_:
            out = out.reindex(columns=self.output_columns_, fill_value=0)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: str | Path) -> "NumericPreprocessor":
        with Path(path).open("rb") as file:
            return pickle.load(file)

    @staticmethod
    def _as_category_strings(series: pd.Series) -> pd.Series:
        return series.astype("object").where(series.notna(), MISSING_TOKEN).astype(str)

    @staticmethod
    def _is_boolean_like(series: pd.Series) -> bool:
        non_missing = set(series.dropna().astype("object").unique().tolist())
        return bool(non_missing) and non_missing.issubset(set(BOOLEAN_LIKE_VALUES))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal fraud preprocessing.")
    parser.add_argument("--data-dir", default="data/example_subset")
    parser.add_argument("--output-dir", default="outputs/example_preprocessing")
    parser.add_argument("--max-onehot-cardinality", type=int, default=30)
    args = parser.parse_args()

    loader = FraudDataLoader(args.data_dir)
    train = loader.load_train()
    preprocessor = NumericPreprocessor(max_onehot_cardinality=args.max_onehot_cardinality)
    numeric_train = preprocessor.fit_transform(train)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_train.to_csv(output_dir / "numeric_train.csv", index=False)
    train[["TransactionID", "isFraud", "has_identity"]].to_csv(
        output_dir / "row_metadata.csv", index=False
    )
    preprocessor.save(output_dir / "numeric_preprocessor.pkl")
    print(f"rows={len(train)} numeric_columns={numeric_train.shape[1]}")


if __name__ == "__main__":
    main()
