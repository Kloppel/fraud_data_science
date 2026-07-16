"""Train and compare compact fraud classifiers for workpacket 4.

The module is intentionally self-contained: it loads IEEE-CIS training data,
uses the selected feature artifact when available, fits interpretable rules, a
human-readable decision tree, a small neural-network classifier, and a random
contender baseline, then writes metrics, model artifacts, and Kaggle-shaped
prediction files.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_preprocessing import FraudDataLoader, NumericPreprocessor


TARGET_COLUMN = "isFraud"
ID_COLUMN = "TransactionID"
MISSING_TOKEN = "__missing__"


@dataclass
class ModelTrainingConfig:
    data_dir: str = "data/example_subset"
    output_dir: str = "outputs/example_model_training"
    selected_features_path: str | None = "outputs/example_feature_analysis/selected_features.csv"
    target_column: str = TARGET_COLUMN
    id_columns: list[str] = field(default_factory=lambda: [ID_COLUMN])
    test_fraction: float = 0.25
    random_state: int = 42
    max_rows: int | None = None
    max_features: int = 60
    max_rules: int = 12
    max_tree_depth: int = 3
    hidden_units: int = 12
    learning_rate: float = 0.05
    epochs: int = 250

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "output_dir": self.output_dir,
            "selected_features_path": self.selected_features_path,
            "target_column": self.target_column,
            "id_columns": self.id_columns,
            "test_fraction": self.test_fraction,
            "random_state": self.random_state,
            "max_rows": self.max_rows,
            "max_features": self.max_features,
            "max_rules": self.max_rules,
            "max_tree_depth": self.max_tree_depth,
            "hidden_units": self.hidden_units,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
        }


class RuleBasedFraudClassifier:
    """Interpretable fraud scorer built from high-risk thresholds and categories."""

    def __init__(self, max_rules: int = 12, min_support: int = 2) -> None:
        self.max_rules = max_rules
        self.min_support = min_support
        self.rules_: list[dict[str, Any]] = []
        self.base_rate_: float = 0.0

    def fit(self, df: pd.DataFrame, features: list[str], target_column: str) -> "RuleBasedFraudClassifier":
        y = df[target_column].astype(int)
        self.base_rate_ = float(y.mean())
        candidates: list[dict[str, Any]] = []
        for feature in features:
            if feature not in df:
                continue
            series = df[feature]
            if pd.api.types.is_numeric_dtype(series):
                candidates.extend(self._numeric_rules(series, y, feature))
            else:
                candidates.extend(self._categorical_rules(series, y, feature))
        candidates.sort(key=lambda rule: (rule["precision"], rule["lift"], rule["support"]), reverse=True)
        self.rules_ = candidates[: self.max_rules]
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        scores = np.full(len(df), self.base_rate_, dtype=float)
        weights = np.zeros(len(df), dtype=float)
        for rule in self.rules_:
            mask = self._rule_mask(df, rule).to_numpy(dtype=bool)
            weight = max(0.01, float(rule["precision"]) - self.base_rate_)
            scores[mask] += weight * float(rule["precision"])
            weights[mask] += weight
        adjusted = np.where(weights > 0, scores / (1.0 + weights), scores)
        return np.clip(adjusted, 0.0, 1.0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self, file)

    def rules_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rules_)

    def _numeric_rules(self, series: pd.Series, y: pd.Series, feature: str) -> list[dict[str, Any]]:
        numeric = pd.to_numeric(series, errors="coerce")
        valid = numeric.notna()
        if valid.sum() < 4 or numeric[valid].nunique() <= 1:
            return []
        fraud_values = numeric[valid & (y == 1)]
        nonfraud_values = numeric[valid & (y == 0)]
        if fraud_values.empty or nonfraud_values.empty:
            return []
        direction = ">=" if fraud_values.median() >= nonfraud_values.median() else "<="
        thresholds = numeric[valid].quantile([0.25, 0.5, 0.75]).dropna().unique()
        rows = []
        for threshold in thresholds:
            rule = self._build_rule(feature, "numeric", direction, float(threshold), numeric, y)
            if rule:
                rows.append(rule)
        return rows

    def _categorical_rules(self, series: pd.Series, y: pd.Series, feature: str) -> list[dict[str, Any]]:
        filled = fill_category(series)
        rows = []
        for value, support in filled.value_counts(dropna=False).items():
            if support < self.min_support:
                continue
            mask = filled == value
            precision = float(y[mask].mean())
            lift = precision / self.base_rate_ if self.base_rate_ else 0.0
            if precision > self.base_rate_ and lift >= 1.25:
                rows.append(
                    {
                        "feature": feature,
                        "kind": "categorical",
                        "operator": "==",
                        "value": str(value),
                        "support": int(support),
                        "precision": precision,
                        "lift": lift,
                    }
                )
        return rows

    def _build_rule(
        self,
        feature: str,
        kind: str,
        operator: str,
        value: float,
        series: pd.Series,
        y: pd.Series,
    ) -> dict[str, Any] | None:
        mask = series >= value if operator == ">=" else series <= value
        support = int(mask.sum())
        if support < self.min_support:
            return None
        precision = float(y[mask].mean())
        lift = precision / self.base_rate_ if self.base_rate_ else 0.0
        if precision <= self.base_rate_ or lift < 1.15:
            return None
        return {
            "feature": feature,
            "kind": kind,
            "operator": operator,
            "value": value,
            "support": support,
            "precision": precision,
            "lift": lift,
        }

    @staticmethod
    def _rule_mask(df: pd.DataFrame, rule: dict[str, Any]) -> pd.Series:
        feature = rule["feature"]
        if feature not in df:
            return pd.Series(False, index=df.index)
        if rule["kind"] == "numeric":
            values = pd.to_numeric(df[feature], errors="coerce")
            return values >= rule["value"] if rule["operator"] == ">=" else values <= rule["value"]
        return fill_category(df[feature]) == str(rule["value"])


class HumanDecisionTreeFraudClassifier:
    """Small binary decision tree with plain-language threshold/equality rules."""

    def __init__(
        self,
        max_depth: int = 3,
        min_leaf_size: int = 2,
        max_category_values: int = 12,
    ) -> None:
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.max_category_values = max_category_values
        self.tree_: dict[str, Any] | None = None
        self.features_: list[str] = []

    def fit(self, df: pd.DataFrame, features: list[str], target_column: str) -> "HumanDecisionTreeFraudClassifier":
        self.features_ = [feature for feature in features if feature in df.columns]
        y = df[target_column].astype(int).reset_index(drop=True)
        x = df[self.features_].reset_index(drop=True)
        weights = pd.Series(balanced_sample_weights(y), index=x.index)
        self.tree_ = self._build_node(x, y, weights, depth=0)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.tree_ is None:
            raise ValueError("Model is not fitted.")
        rows = df.reset_index(drop=True)
        return np.array([self._predict_row(row, self.tree_) for _, row in rows.iterrows()])

    def rules_frame(self) -> pd.DataFrame:
        if self.tree_ is None:
            return pd.DataFrame(columns=["path", "prediction", "samples", "fraud_rate"])
        rows: list[dict[str, Any]] = []
        self._collect_rules(self.tree_, [], rows)
        return pd.DataFrame(rows)

    def rules_markdown(self) -> str:
        rules = self.rules_frame()
        lines = ["# Human Decision Tree Rules", ""]
        if rules.empty:
            lines.append("_No rules._")
            return "\n".join(lines)
        for idx, row in rules.iterrows():
            lines.append(f"## Leaf {idx + 1}")
            lines.append("")
            lines.append(f"- If: {row['path']}")
            lines.append(f"- Predicted fraud probability: {row['prediction']:.4f}")
            lines.append(f"- Training samples: {int(row['samples'])}")
            lines.append(f"- Training fraud rate: {row['fraud_rate']:.4f}")
            lines.append("")
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self, file)

    def _build_node(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        weights: pd.Series,
        depth: int,
    ) -> dict[str, Any]:
        node = self._leaf_node(y, weights)
        if depth >= self.max_depth or len(x) < 2 * self.min_leaf_size or y.nunique() <= 1:
            return node
        split = self._best_split(x, y, weights)
        if split is None:
            return node
        left_mask = split["mask"]
        right_mask = ~left_mask
        node.update(
            {
                "is_leaf": False,
                "split": {key: value for key, value in split.items() if key != "mask"},
                "left": self._build_node(
                    x[left_mask].reset_index(drop=True),
                    y[left_mask].reset_index(drop=True),
                    weights[left_mask].reset_index(drop=True),
                    depth + 1,
                ),
                "right": self._build_node(
                    x[right_mask].reset_index(drop=True),
                    y[right_mask].reset_index(drop=True),
                    weights[right_mask].reset_index(drop=True),
                    depth + 1,
                ),
            }
        )
        return node

    def _best_split(
        self, x: pd.DataFrame, y: pd.Series, weights: pd.Series
    ) -> dict[str, Any] | None:
        parent_impurity = weighted_gini(y, weights)
        best: dict[str, Any] | None = None
        for feature in self.features_:
            if feature not in x:
                continue
            for candidate in self._split_candidates(x[feature], feature):
                mask = self._candidate_mask(x[feature], candidate)
                left_n = int(mask.sum())
                right_n = int((~mask).sum())
                if left_n < self.min_leaf_size or right_n < self.min_leaf_size:
                    continue
                left_weight = float(weights[mask].sum())
                right_weight = float(weights[~mask].sum())
                total_weight = left_weight + right_weight
                impurity = (
                    left_weight / total_weight * weighted_gini(y[mask], weights[mask])
                    + right_weight / total_weight * weighted_gini(y[~mask], weights[~mask])
                )
                gain = parent_impurity - impurity
                if best is None or gain > best["gain"]:
                    best = {**candidate, "mask": mask, "gain": float(gain)}
        if best is None or best["gain"] <= 0:
            return None
        return best

    def _split_candidates(self, series: pd.Series, feature: str) -> list[dict[str, Any]]:
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            thresholds = numeric.dropna().quantile([0.25, 0.5, 0.75]).dropna().unique()
            return [
                {
                    "feature": feature,
                    "kind": "numeric",
                    "operator": "<=",
                    "value": float(threshold),
                }
                for threshold in thresholds
            ]
        filled = fill_category(series)
        values = filled.value_counts().head(self.max_category_values).index.tolist()
        return [
            {
                "feature": feature,
                "kind": "categorical",
                "operator": "==",
                "value": str(value),
            }
            for value in values
        ]

    @staticmethod
    def _candidate_mask(series: pd.Series, candidate: dict[str, Any]) -> pd.Series:
        if candidate["kind"] == "numeric":
            return pd.to_numeric(series, errors="coerce") <= float(candidate["value"])
        return fill_category(series) == str(candidate["value"])

    @staticmethod
    def _leaf_node(y: pd.Series, weights: pd.Series) -> dict[str, Any]:
        weighted_positive = float(weights[y == 1].sum())
        weighted_total = float(weights.sum())
        return {
            "is_leaf": True,
            "prediction": safe_divide(weighted_positive, weighted_total),
            "samples": int(len(y)),
            "fraud_rate": float(y.mean()) if len(y) else 0.0,
        }

    def _predict_row(self, row: pd.Series, node: dict[str, Any]) -> float:
        if node["is_leaf"]:
            return float(node["prediction"])
        split = node["split"]
        if split["kind"] == "numeric":
            value = pd.to_numeric(pd.Series([row.get(split["feature"])]), errors="coerce").iloc[0]
            go_left = pd.notna(value) and float(value) <= float(split["value"])
        else:
            go_left = str(row.get(split["feature"], MISSING_TOKEN)) == str(split["value"])
        return self._predict_row(row, node["left"] if go_left else node["right"])

    def _collect_rules(
        self,
        node: dict[str, Any],
        path: list[str],
        rows: list[dict[str, Any]],
    ) -> None:
        if node["is_leaf"]:
            rows.append(
                {
                    "path": " AND ".join(path) if path else "always",
                    "prediction": float(node["prediction"]),
                    "samples": int(node["samples"]),
                    "fraud_rate": float(node["fraud_rate"]),
                }
            )
            return
        split = node["split"]
        condition = format_tree_condition(split, positive=True)
        inverse = format_tree_condition(split, positive=False)
        self._collect_rules(node["left"], path + [condition], rows)
        self._collect_rules(node["right"], path + [inverse], rows)


class NeuralNetworkFraudClassifier:
    """Small one-hidden-layer neural network trained with balanced sample weights."""

    def __init__(
        self,
        hidden_units: int = 12,
        learning_rate: float = 0.05,
        epochs: int = 250,
        random_state: int = 42,
    ) -> None:
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.w1_: np.ndarray | None = None
        self.b1_: np.ndarray | None = None
        self.w2_: np.ndarray | None = None
        self.b2_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NeuralNetworkFraudClassifier":
        x = X.to_numpy(dtype=float)
        target = y.to_numpy(dtype=float).reshape(-1, 1)
        rng = np.random.default_rng(self.random_state)
        scale = 1.0 / math.sqrt(max(1, x.shape[1]))
        self.w1_ = rng.normal(0.0, scale, size=(x.shape[1], self.hidden_units))
        self.b1_ = np.zeros((1, self.hidden_units))
        self.w2_ = rng.normal(0.0, scale, size=(self.hidden_units, 1))
        self.b2_ = 0.0
        weights = balanced_sample_weights(y).reshape(-1, 1)
        normalizer = max(float(weights.sum()), 1.0)

        for _ in range(self.epochs):
            hidden_linear = x @ self.w1_ + self.b1_
            hidden = np.tanh(hidden_linear)
            logits = hidden @ self.w2_ + self.b2_
            proba = sigmoid(logits)
            error = (proba - target) * weights
            grad_w2 = hidden.T @ error / normalizer
            grad_b2 = float(error.sum() / normalizer)
            hidden_error = (error @ self.w2_.T) * (1.0 - hidden**2)
            grad_w1 = x.T @ hidden_error / normalizer
            grad_b1 = hidden_error.sum(axis=0, keepdims=True) / normalizer
            self.w2_ -= self.learning_rate * grad_w2
            self.b2_ -= self.learning_rate * grad_b2
            self.w1_ -= self.learning_rate * grad_w1
            self.b1_ -= self.learning_rate * grad_b1
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.w1_ is None or self.b1_ is None or self.w2_ is None:
            raise ValueError("Model is not fitted.")
        x = X.to_numpy(dtype=float)
        hidden = np.tanh(x @ self.w1_ + self.b1_)
        return sigmoid(hidden @ self.w2_ + self.b2_).ravel()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self, file)


class RandomContenderClassifier:
    """Seeded random baseline centered around the training fraud prior."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.base_rate_: float = 0.0

    def fit(self, y: pd.Series) -> "RandomContenderClassifier":
        self.base_rate_ = float(y.mean())
        return self

    def predict_proba(self, n_rows: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        noise = rng.beta(2, 5, size=n_rows)
        return np.clip(0.7 * self.base_rate_ + 0.3 * noise, 0.0, 1.0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self, file)


def run_model_training(config: ModelTrainingConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = FraudDataLoader(config.data_dir).load_train(nrows=config.max_rows)
    validate_two_class_target(df[config.target_column])
    train_df, valid_df = stratified_split(
        df, config.target_column, config.test_fraction, config.random_state
    )
    features = choose_features(train_df, config)
    bundle = train_model_bundle(train_df, valid_df, features, config, output_dir)
    test_submission_written = write_test_submission_if_available(
        features, config, output_dir, bundle["preprocessor"], bundle["neural_model"]
    )
    outputs = [
        "model_comparison_metrics.csv",
        "confusion_matrices.csv",
        "rules_based_model.pkl",
        "decision_tree_model.pkl",
        "neural_network_model.pkl",
        "random_contender_model.pkl",
        "numeric_preprocessor.pkl",
        "validation_predictions.csv",
        "validation_submission.csv",
        "interpretable_rules.csv",
        "decision_tree_rules.csv",
        "decision_tree_rules.md",
    ]
    if test_submission_written:
        outputs.append("test_submission.csv")
    summary = {
        "rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "features_used": int(len(features)),
        "test_submission_written": test_submission_written,
        "outputs": outputs,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def train_model_bundle(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    features: list[str],
    config: ModelTrainingConfig,
    output_dir: Path,
) -> dict[str, Any]:
    preprocessor = NumericPreprocessor(
        exclude_columns=config.id_columns + [config.target_column],
        max_onehot_cardinality=20,
    )
    train_numeric = preprocessor.fit_transform(train_df[features + [config.target_column] + config.id_columns])
    valid_numeric = preprocessor.transform(valid_df[features + [config.target_column] + config.id_columns])
    y_train = train_df[config.target_column].astype(int)
    y_valid = valid_df[config.target_column].astype(int)

    rules = RuleBasedFraudClassifier(max_rules=config.max_rules).fit(
        train_df, features, config.target_column
    )
    decision_tree = HumanDecisionTreeFraudClassifier(max_depth=config.max_tree_depth).fit(
        train_df, features, config.target_column
    )
    neural = NeuralNetworkFraudClassifier(
        hidden_units=config.hidden_units,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        random_state=config.random_state,
    ).fit(train_numeric, y_train)
    random_model = RandomContenderClassifier(config.random_state).fit(y_train)

    predictions = pd.DataFrame(
        {
            ID_COLUMN: valid_df[ID_COLUMN].to_numpy(),
            "actual": y_valid.to_numpy(),
            "rules_based": rules.predict_proba(valid_df),
            "decision_tree": decision_tree.predict_proba(valid_df),
            "neural_network": neural.predict_proba(valid_numeric),
            "random_contender": random_model.predict_proba(len(valid_df)),
        }
    )
    metrics, confusion = compare_models(predictions, "actual")
    metrics.to_csv(output_dir / "model_comparison_metrics.csv", index=False)
    confusion.to_csv(output_dir / "confusion_matrices.csv", index=False)
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    write_submission(predictions[[ID_COLUMN, "neural_network"]], output_dir / "validation_submission.csv")
    rules.rules_frame().to_csv(output_dir / "interpretable_rules.csv", index=False)
    decision_tree.rules_frame().to_csv(output_dir / "decision_tree_rules.csv", index=False)
    (output_dir / "decision_tree_rules.md").write_text(decision_tree.rules_markdown())
    rules.save(output_dir / "rules_based_model.pkl")
    decision_tree.save(output_dir / "decision_tree_model.pkl")
    neural.save(output_dir / "neural_network_model.pkl")
    random_model.save(output_dir / "random_contender_model.pkl")
    preprocessor.save(output_dir / "numeric_preprocessor.pkl")
    return {
        "metrics": metrics,
        "confusion": confusion,
        "predictions": predictions,
        "preprocessor": preprocessor,
        "neural_model": neural,
        "decision_tree_model": decision_tree,
    }


def write_test_submission_if_available(
    features: list[str],
    config: ModelTrainingConfig,
    output_dir: Path,
    preprocessor: NumericPreprocessor,
    neural_model: NeuralNetworkFraudClassifier,
) -> bool:
    try:
        test_df = FraudDataLoader(config.data_dir).load_test(nrows=config.max_rows)
    except FileNotFoundError:
        return False
    missing = [feature for feature in features if feature not in test_df.columns]
    if missing:
        return False
    test_numeric = preprocessor.transform(test_df[features + config.id_columns])
    test_predictions = pd.DataFrame(
        {
            ID_COLUMN: test_df[ID_COLUMN].to_numpy(),
            "neural_network": neural_model.predict_proba(test_numeric),
        }
    )
    write_submission(test_predictions, output_dir / "test_submission.csv")
    return True


def choose_features(df: pd.DataFrame, config: ModelTrainingConfig) -> list[str]:
    excluded = set(config.id_columns + [config.target_column])
    selected = load_selected_features(config.selected_features_path)
    features = [feature for feature in selected if feature in df.columns and feature not in excluded]
    if not features:
        candidate_columns = [column for column in df.columns if column not in excluded]
        scores = []
        y = df[config.target_column].astype(int)
        for column in candidate_columns:
            scores.append((column, quick_feature_power(df[column], y)))
        scores.sort(key=lambda item: item[1], reverse=True)
        features = [column for column, _ in scores[: config.max_features]]
    return features[: config.max_features]


def load_selected_features(path: str | None) -> list[str]:
    if not path:
        return []
    selected_path = Path(path)
    if not selected_path.exists():
        return []
    frame = pd.read_csv(selected_path)
    if "selected" in frame.columns:
        frame = frame[frame["selected"].astype(str).str.lower().isin(["true", "1"])]
    if "feature" not in frame.columns:
        return []
    return frame["feature"].dropna().astype(str).tolist()


def compare_models(predictions: pd.DataFrame, actual_column: str = "actual") -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true = predictions[actual_column].astype(int)
    metric_rows = []
    confusion_rows = []
    for model_name in [c for c in predictions.columns if c not in {ID_COLUMN, actual_column}]:
        scores = predictions[model_name].astype(float)
        labels = (scores >= 0.5).astype(int)
        counts = confusion_counts(y_true, labels)
        metric_rows.append(
            {
                "model": model_name,
                "roc_auc": roc_auc_score(y_true, scores),
                "pr_auc": average_precision_score(y_true, scores),
                "precision": safe_divide(counts["tp"], counts["tp"] + counts["fp"]),
                "recall": safe_divide(counts["tp"], counts["tp"] + counts["fn"]),
                "f1": f1_score(counts),
            }
        )
        confusion_rows.append({"model": model_name, **counts})
    return pd.DataFrame(metric_rows), pd.DataFrame(confusion_rows)


def write_submission(predictions: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    submission = predictions.rename(columns={predictions.columns[1]: TARGET_COLUMN})[
        [ID_COLUMN, TARGET_COLUMN]
    ]
    submission.to_csv(path, index=False)


def stratified_split(
    df: pd.DataFrame,
    target_column: str,
    test_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_parts = []
    for _, group in df.groupby(target_column, sort=False):
        if len(group) < 2:
            raise ValueError("Each class needs at least two rows for a validation split.")
        n_valid = max(1, int(round(len(group) * test_fraction)))
        valid_parts.append(group.sample(n=min(n_valid, len(group) - 1), random_state=random_state))
    valid = pd.concat(valid_parts).sort_index()
    train = df.drop(index=valid.index)
    return train.reset_index(drop=True), valid.reset_index(drop=True)


def validate_two_class_target(target: pd.Series) -> None:
    labels = set(target.dropna().astype(int).unique().tolist())
    if labels != {0, 1}:
        raise ValueError(f"Training requires both target classes [0, 1], found {sorted(labels)!r}.")


def quick_feature_power(series: pd.Series, y: pd.Series) -> float:
    if pd.api.types.is_numeric_dtype(series):
        scores = pd.to_numeric(series, errors="coerce")
    else:
        filled = fill_category(series)
        rates = y.groupby(filled).mean()
        scores = filled.map(rates)
    return abs(roc_auc_score(y, scores) - 0.5)


def roc_auc_score(y_true: pd.Series, scores: pd.Series | np.ndarray) -> float:
    y = pd.Series(y_true).astype(int).reset_index(drop=True)
    x = pd.Series(scores).astype(float).reset_index(drop=True)
    valid = x.notna() & y.notna()
    y = y[valid]
    x = x[valid]
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = x.rank(method="average")
    rank_sum_positive = float(ranks[y == 1].sum())
    return float((rank_sum_positive - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision_score(y_true: pd.Series, scores: pd.Series | np.ndarray) -> float:
    frame = pd.DataFrame({"y": pd.Series(y_true).astype(int), "score": pd.Series(scores).astype(float)})
    frame = frame.dropna().sort_values("score", ascending=False)
    positives = int((frame["y"] == 1).sum())
    if positives == 0:
        return math.nan
    tp = 0
    fp = 0
    precision_sum = 0.0
    for _, row in frame.iterrows():
        if int(row["y"]) == 1:
            tp += 1
            precision_sum += tp / (tp + fp)
        else:
            fp += 1
    return float(precision_sum / positives)


def confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> dict[str, int]:
    y = y_true.astype(int)
    pred = y_pred.astype(int)
    return {
        "tn": int(((y == 0) & (pred == 0)).sum()),
        "fp": int(((y == 0) & (pred == 1)).sum()),
        "fn": int(((y == 1) & (pred == 0)).sum()),
        "tp": int(((y == 1) & (pred == 1)).sum()),
    }


def f1_score(counts: dict[str, int]) -> float:
    precision = safe_divide(counts["tp"], counts["tp"] + counts["fp"])
    recall = safe_divide(counts["tp"], counts["tp"] + counts["fn"])
    return safe_divide(2 * precision * recall, precision + recall)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def balanced_sample_weights(y: pd.Series) -> np.ndarray:
    values = y.astype(int).to_numpy()
    weights = np.ones(len(values), dtype=float)
    for label in (0, 1):
        mask = values == label
        if mask.any():
            weights[mask] = len(values) / (2.0 * mask.sum())
    return weights


def weighted_gini(y: pd.Series, weights: pd.Series) -> float:
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    positive_weight = float(weights[y.astype(int) == 1].sum())
    p_positive = positive_weight / total
    p_negative = 1.0 - p_positive
    return 1.0 - p_positive**2 - p_negative**2


def format_tree_condition(split: dict[str, Any], positive: bool) -> str:
    feature = split["feature"]
    value = split["value"]
    if split["kind"] == "numeric":
        operator = "<=" if positive else ">"
        return f"{feature} {operator} {float(value):.6g}"
    operator = "==" if positive else "!="
    return f"{feature} {operator} {value}"


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-values))


def fill_category(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), MISSING_TOKEN).astype(str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train three fraud classifiers.")
    parser.add_argument("--data-dir", default="data/example_subset")
    parser.add_argument("--output-dir", default="outputs/example_model_training")
    parser.add_argument("--selected-features-path", default="outputs/example_feature_analysis/selected_features.csv")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-features", type=int, default=60)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-rules", type=int, default=12)
    parser.add_argument("--max-tree-depth", type=int, default=3)
    parser.add_argument("--hidden-units", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ModelTrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        selected_features_path=args.selected_features_path,
        max_rows=args.max_rows,
        max_features=args.max_features,
        test_fraction=args.test_fraction,
        random_state=args.random_state,
        max_rules=args.max_rules,
        max_tree_depth=args.max_tree_depth,
        hidden_units=args.hidden_units,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )
    result = run_model_training(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
