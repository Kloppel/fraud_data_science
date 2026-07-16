"""Reusable binary feature analysis and selection for IEEE-CIS fraud data.

The module is intentionally self-contained for workpacket 3: it loads a small
train split, analyzes feature signal without looking at the held-out rows,
evaluates selected subsets on held-out rows, and writes compact artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_preprocessing import FraudDataLoader


MISSING_TOKEN = "__missing__"
BOOLEAN_MAP = {"T": 1, "F": 0, True: 1, False: 0, 1: 1, 0: 0}


@dataclass
class FeatureAnalysisConfig:
    """Runtime options for leakage-safe binary feature analysis."""

    data_dir: str = "data/example_subset"
    output_dir: str = "outputs/example_feature_analysis"
    target_column: str = "isFraud"
    id_columns: list[str] = field(default_factory=lambda: ["TransactionID"])
    group_columns: list[str] = field(default_factory=list)
    excluded_columns: list[str] = field(default_factory=list)
    test_fraction: float = 0.25
    random_state: int = 42
    max_rows: int | None = None
    top_n: int = 30
    min_prediction_power: float = 0.02
    redundancy_threshold: float = 0.95
    stability_threshold: float = 0.15
    near_constant_threshold: float = 0.995
    strategies: list[str] = field(default_factory=lambda: ["conservative", "balanced"])

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "FeatureAnalysisConfig":
        known = {field_name for field_name in cls.__dataclass_fields__}
        kwargs = {key: value for key, value in values.items() if key in known}
        for key in ("id_columns", "group_columns", "excluded_columns", "strategies"):
            if key in kwargs and isinstance(kwargs[key], str):
                kwargs[key] = [part.strip() for part in kwargs[key].split(",") if part.strip()]
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "output_dir": self.output_dir,
            "target_column": self.target_column,
            "id_columns": self.id_columns,
            "group_columns": self.group_columns,
            "excluded_columns": self.excluded_columns,
            "test_fraction": self.test_fraction,
            "random_state": self.random_state,
            "max_rows": self.max_rows,
            "top_n": self.top_n,
            "min_prediction_power": self.min_prediction_power,
            "redundancy_threshold": self.redundancy_threshold,
            "stability_threshold": self.stability_threshold,
            "near_constant_threshold": self.near_constant_threshold,
            "strategies": self.strategies,
        }


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load JSON or simple YAML config without adding a hard YAML dependency."""
    if path is None:
        return {}
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded or {}
    except ImportError:
        return _parse_simple_yaml(text)


def merge_config(
    config_path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> FeatureAnalysisConfig:
    values = load_config(config_path)
    for key, value in (overrides or {}).items():
        if value is not None:
            values[key] = value
    return FeatureAnalysisConfig.from_mapping(values)


def run_feature_analysis(config: FeatureAnalysisConfig) -> dict[str, Any]:
    """Run leakage-safe feature analysis and write reports/artifacts."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = FraudDataLoader(config.data_dir).load_train(nrows=config.max_rows)
    validate_binary_target(df[config.target_column], allow_single_class=True)
    train_df, heldout_df = train_heldout_split(
        df,
        target_column=config.target_column,
        test_fraction=config.test_fraction,
        random_state=config.random_state,
    )

    feature_columns = select_input_columns(train_df, config)
    audit = audit_features(train_df, feature_columns, config.target_column)
    if has_two_classes(train_df[config.target_column]):
        train_ranking = rank_features(train_df, feature_columns, config.target_column)
        stability = feature_stability(
            train_df, heldout_df, feature_columns, config.target_column
        )
        redundancy_pairs, redundancy_clusters = redundancy_analysis(
            train_df, feature_columns, threshold=config.redundancy_threshold
        )
        model_importance = model_based_importance(
            train_df, feature_columns, config.target_column, config.random_state
        )
        permutation = class_specific_permutation_importance(
            train_df,
            heldout_df,
            feature_columns,
            config.target_column,
            config.random_state,
        )
        composite = composite_ranking(
            audit, train_ranking, stability, model_importance, permutation
        )
        selected = select_features(
            composite,
            audit,
            redundancy_pairs,
            min_prediction_power=config.min_prediction_power,
            stability_threshold=config.stability_threshold,
        )
        subset_metrics = evaluate_feature_subsets(
            train_df,
            heldout_df,
            selected,
            composite,
            config.target_column,
            strategies=config.strategies,
        )
        report = feature_report(
            config, train_df, heldout_df, composite, selected, subset_metrics
        )
    else:
        train_ranking = empty_ranking()
        stability = empty_stability()
        redundancy_pairs, redundancy_clusters = redundancy_analysis(
            train_df, feature_columns, threshold=config.redundancy_threshold
        )
        model_importance = empty_model_importance()
        permutation = empty_permutation()
        composite = empty_composite()
        selected = pd.DataFrame(columns=["feature", "drop_reason", "selected"])
        subset_metrics = pd.DataFrame(columns=["strategy", "n_features", "heldout_auc"])
        report = (
            "# Feature Analysis Report\n\n"
            "Binary feature analysis was skipped because the train split does not "
            "contain both target classes.\n"
        )

    audit.to_csv(output_dir / "feature_audit.csv", index=False)
    train_ranking.to_csv(output_dir / "train_only_feature_ranking.csv", index=False)
    stability.to_csv(output_dir / "feature_stability.csv", index=False)
    redundancy_pairs.to_csv(output_dir / "redundancy_pairs.csv", index=False)
    redundancy_clusters.to_csv(output_dir / "redundancy_clusters.csv", index=False)
    model_importance.to_csv(output_dir / "model_importance.csv", index=False)
    permutation.to_csv(output_dir / "class_specific_permutation_importance.csv", index=False)
    composite.to_csv(output_dir / "composite_feature_ranking.csv", index=False)
    selected.to_csv(output_dir / "selected_features.csv", index=False)
    subset_metrics.to_csv(output_dir / "feature_subset_metrics.csv", index=False)
    (output_dir / "feature_analysis_report.md").write_text(report)
    write_bar_svg(composite.head(config.top_n), output_dir / "composite_ranking.svg")

    pipeline = {
        "config": config.to_dict(),
        "selected_features": selected.loc[selected["selected"], "feature"].tolist()
        if "selected" in selected
        else [],
        "feature_columns": feature_columns,
    }
    with (output_dir / "feature_selection_pipeline.pkl").open("wb") as file:
        pickle.dump(pipeline, file)

    summary = {
        "rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "heldout_rows": int(len(heldout_df)),
        "features_analyzed": int(len(feature_columns)),
        "selected_features": int(selected["selected"].sum()) if "selected" in selected else 0,
        "has_two_classes": has_two_classes(train_df[config.target_column]),
        "outputs": [
            "feature_audit.csv",
            "train_only_feature_ranking.csv",
            "feature_stability.csv",
            "redundancy_pairs.csv",
            "redundancy_clusters.csv",
            "model_importance.csv",
            "class_specific_permutation_importance.csv",
            "composite_feature_ranking.csv",
            "selected_features.csv",
            "feature_subset_metrics.csv",
            "feature_analysis_report.md",
            "composite_ranking.svg",
            "feature_selection_pipeline.pkl",
        ],
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def select_input_columns(df: pd.DataFrame, config: FeatureAnalysisConfig) -> list[str]:
    excluded = (
        set(config.id_columns)
        | set(config.group_columns)
        | set(config.excluded_columns)
        | {config.target_column}
    )
    return [column for column in df.columns if column not in excluded]


def validate_binary_target(target: pd.Series, allow_single_class: bool = False) -> pd.Series:
    labels = set(target.dropna().unique().tolist())
    if allow_single_class and labels in ({0}, {1}, {0, 1}):
        return target.astype(int)
    if labels != {0, 1}:
        raise ValueError(f"Expected binary labels [0, 1], found {sorted(labels)!r}")
    return target.astype(int)


def train_heldout_split(
    df: pd.DataFrame,
    target_column: str = "isFraud",
    test_fraction: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic split, stratifying when both classes are present."""
    if len(df) <= 1 or test_fraction <= 0:
        return df.copy(), df.head(0).copy()
    rng_state = int(random_state)
    parts = []
    if has_two_classes(df[target_column]):
        for _, group in df.groupby(target_column, sort=False):
            heldout_n = max(1, int(round(len(group) * test_fraction)))
            parts.append(group.sample(n=min(heldout_n, len(group) - 1), random_state=rng_state))
        heldout = pd.concat(parts).sort_index()
    else:
        heldout_n = max(1, int(round(len(df) * test_fraction)))
        heldout = df.sample(n=min(heldout_n, len(df) - 1), random_state=rng_state).sort_index()
    train = df.drop(index=heldout.index)
    return train.reset_index(drop=True), heldout.reset_index(drop=True)


def audit_features(
    df: pd.DataFrame, feature_columns: list[str], target_column: str = "isFraud"
) -> pd.DataFrame:
    rows = []
    for feature in feature_columns:
        series = df[feature]
        frequencies = series.value_counts(normalize=True, dropna=False)
        n_unique = int(series.nunique(dropna=True))
        rows.append(
            {
                "feature": feature,
                "feature_type": infer_feature_type(series),
                "missing_fraction": float(series.isna().mean()),
                "n_unique": n_unique,
                "most_frequent_fraction": float(frequencies.iloc[0]) if len(frequencies) else math.nan,
                "is_constant": n_unique <= 1,
                "is_near_constant": bool(
                    len(frequencies) and frequencies.iloc[0] >= 0.995
                ),
                "is_likely_identifier": bool(
                    len(df) and n_unique / len(df) >= 0.98 and n_unique > 10
                ),
            }
        )
    return pd.DataFrame(rows)


def rank_features(
    df: pd.DataFrame, feature_columns: list[str], target_column: str = "isFraud"
) -> pd.DataFrame:
    y = validate_binary_target(df[target_column])
    rows = []
    for feature in feature_columns:
        score = feature_score(df[feature], y)
        rows.append(
            {
                "feature": feature,
                "signed_auc": score,
                "prediction_power": abs(score) if pd.notna(score) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "prediction_power", ascending=False
    ).reset_index(drop=True)


def feature_stability(
    train_df: pd.DataFrame,
    heldout_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "isFraud",
) -> pd.DataFrame:
    if heldout_df.empty or not has_two_classes(heldout_df[target_column]):
        return pd.DataFrame(
            [
                {
                    "feature": feature,
                    "train_signed_auc": feature_score(
                        train_df[feature], train_df[target_column]
                    ),
                    "heldout_signed_auc": math.nan,
                    "abs_auc_delta": math.nan,
                    "stable": True,
                }
                for feature in feature_columns
            ]
        )
    rows = []
    for feature in feature_columns:
        train_auc = feature_score(train_df[feature], train_df[target_column])
        heldout_auc = feature_score(heldout_df[feature], heldout_df[target_column])
        delta = abs(_nan_to_zero(train_auc) - _nan_to_zero(heldout_auc))
        rows.append(
            {
                "feature": feature,
                "train_signed_auc": train_auc,
                "heldout_signed_auc": heldout_auc,
                "abs_auc_delta": delta,
                "stable": bool(delta <= 0.15),
            }
        )
    return pd.DataFrame(rows)


def redundancy_analysis(
    df: pd.DataFrame, feature_columns: list[str], threshold: float = 0.95
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = model_matrix(df, feature_columns, max_levels=12)
    if matrix.shape[1] < 2:
        empty_pairs = pd.DataFrame(columns=["feature_a", "feature_b", "correlation"])
        empty_clusters = pd.DataFrame(columns=["cluster_id", "feature", "cluster_size"])
        return empty_pairs, empty_clusters
    corr = matrix.corr().fillna(0.0)
    pairs = []
    for idx, feature_a in enumerate(corr.columns):
        for feature_b in corr.columns[idx + 1 :]:
            value = float(corr.loc[feature_a, feature_b])
            if abs(value) >= threshold:
                pairs.append(
                    {
                        "feature_a": root_feature(feature_a),
                        "feature_b": root_feature(feature_b),
                        "encoded_feature_a": feature_a,
                        "encoded_feature_b": feature_b,
                        "correlation": value,
                    }
                )
    pair_df = pd.DataFrame(pairs).drop_duplicates(
        subset=["feature_a", "feature_b"]
    ) if pairs else pd.DataFrame(
        columns=["feature_a", "feature_b", "encoded_feature_a", "encoded_feature_b", "correlation"]
    )
    cluster_df = redundancy_clusters(pair_df)
    return pair_df, cluster_df


def redundancy_clusters(pair_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame(columns=["cluster_id", "feature", "cluster_size"])
    graph: dict[str, set[str]] = {}
    for _, row in pair_df.iterrows():
        a = row["feature_a"]
        b = row["feature_b"]
        if a == b:
            continue
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    seen = set()
    rows = []
    cluster_id = 0
    for start in sorted(graph):
        if start in seen:
            continue
        stack = [start]
        component = []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            stack.extend(graph.get(node, set()) - seen)
        if len(component) > 1:
            cluster_id += 1
            for feature in sorted(component):
                rows.append(
                    {
                        "cluster_id": cluster_id,
                        "feature": feature,
                        "cluster_size": len(component),
                    }
                )
    return pd.DataFrame(rows, columns=["cluster_id", "feature", "cluster_size"])


def model_based_importance(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "isFraud",
    random_state: int = 42,
) -> pd.DataFrame:
    X = model_matrix(df, feature_columns)
    if X.empty:
        return empty_model_importance()
    y = validate_binary_target(df[target_column])
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
        model.fit(X, y)
        encoded = pd.DataFrame(
            {"encoded_feature": X.columns, "importance": abs(model.coef_.ravel())}
        )
        encoded["feature"] = encoded["encoded_feature"].map(root_feature)
        return (
            encoded.groupby("feature", as_index=False)["importance"]
            .max()
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        ranking = rank_features(df, feature_columns, target_column)
        return ranking.rename(columns={"prediction_power": "importance"})[
            ["feature", "importance"]
        ]


def class_specific_permutation_importance(
    train_df: pd.DataFrame,
    heldout_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "isFraud",
    random_state: int = 42,
) -> pd.DataFrame:
    eval_df = heldout_df if has_two_classes(heldout_df[target_column]) else train_df
    base_scores = {
        feature: feature_score(eval_df[feature], eval_df[target_column])
        for feature in feature_columns
    }
    rows = []
    for feature in feature_columns:
        for cls in (0, 1):
            permuted = eval_df.copy()
            mask = permuted[target_column] == cls
            if mask.sum() > 1:
                permuted.loc[mask, feature] = (
                    permuted.loc[mask, feature]
                    .sample(frac=1.0, random_state=random_state)
                    .to_numpy()
                )
            new_score = feature_score(permuted[feature], permuted[target_column])
            rows.append(
                {
                    "feature": feature,
                    "class_label": cls,
                    "permutation_importance": max(
                        0.0, abs(_nan_to_zero(base_scores[feature])) - abs(_nan_to_zero(new_score))
                    ),
                }
            )
    if not rows:
        return empty_permutation()
    return pd.DataFrame(rows).sort_values(
        "permutation_importance", ascending=False
    ).reset_index(drop=True)


def composite_ranking(
    audit: pd.DataFrame,
    train_ranking: pd.DataFrame,
    stability: pd.DataFrame,
    model_importance: pd.DataFrame,
    permutation: pd.DataFrame,
) -> pd.DataFrame:
    permutation_summary = (
        permutation.groupby("feature", as_index=False)["permutation_importance"].max()
        if not permutation.empty
        else pd.DataFrame(columns=["feature", "permutation_importance"])
    )
    frame = audit.merge(train_ranking, on="feature", how="left")
    frame = frame.merge(stability, on="feature", how="left")
    frame = frame.merge(model_importance, on="feature", how="left")
    frame = frame.merge(permutation_summary, on="feature", how="left")
    for column in ("prediction_power", "importance", "permutation_importance"):
        frame[column] = frame[column].fillna(0.0)
        max_value = frame[column].max()
        frame[f"{column}_rank_score"] = frame[column] / max_value if max_value else 0.0
    penalty = (
        frame["is_constant"].astype(float)
        + frame["is_near_constant"].astype(float)
        + frame["is_likely_identifier"].astype(float)
        + frame["abs_auc_delta"].fillna(0.0).clip(upper=1.0)
    )
    frame["composite_score"] = (
        0.45 * frame["prediction_power_rank_score"]
        + 0.35 * frame["importance_rank_score"]
        + 0.20 * frame["permutation_importance_rank_score"]
        - 0.15 * penalty
    ).clip(lower=0.0)
    return frame.sort_values("composite_score", ascending=False).reset_index(drop=True)


def select_features(
    composite: pd.DataFrame,
    audit: pd.DataFrame,
    redundancy_pairs: pd.DataFrame,
    min_prediction_power: float = 0.02,
    stability_threshold: float = 0.15,
) -> pd.DataFrame:
    redundant_drop = redundant_features_to_drop(composite, redundancy_pairs)
    rows = []
    for _, row in composite.iterrows():
        reasons = []
        if bool(row.get("is_likely_identifier", False)):
            reasons.append("likely_identifier")
        if bool(row.get("is_constant", False)) or bool(row.get("is_near_constant", False)):
            reasons.append("near_constant")
        if row["feature"] in redundant_drop:
            reasons.append("redundant")
        if row.get("abs_auc_delta", 0.0) > stability_threshold:
            reasons.append("unstable")
        if row.get("prediction_power", 0.0) < min_prediction_power:
            reasons.append("weak_prediction_power")
        rows.append(
            {
                "feature": row["feature"],
                "composite_score": row["composite_score"],
                "prediction_power": row.get("prediction_power", 0.0),
                "selected": not reasons,
                "drop_reason": ",".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def evaluate_feature_subsets(
    train_df: pd.DataFrame,
    heldout_df: pd.DataFrame,
    selected: pd.DataFrame,
    composite: pd.DataFrame,
    target_column: str = "isFraud",
    strategies: list[str] | None = None,
) -> pd.DataFrame:
    strategies = strategies or ["conservative", "balanced"]
    rows = []
    chosen = selected.loc[selected["selected"], "feature"].tolist()
    ranked = composite["feature"].tolist()
    for strategy in strategies:
        if strategy == "top_10":
            features = ranked[:10]
        elif strategy == "balanced":
            features = chosen[: max(1, min(25, len(chosen)))]
        else:
            features = chosen
        heldout_auc = subset_auc(train_df, heldout_df, features, target_column)
        rows.append(
            {
                "strategy": strategy,
                "n_features": len(features),
                "features": ",".join(features),
                "heldout_auc": heldout_auc,
            }
        )
    return pd.DataFrame(rows)


def subset_auc(
    train_df: pd.DataFrame,
    heldout_df: pd.DataFrame,
    features: list[str],
    target_column: str = "isFraud",
) -> float:
    if not features or heldout_df.empty or not has_two_classes(heldout_df[target_column]):
        return math.nan
    rates = {}
    global_rate = float(train_df[target_column].mean())
    for feature in features:
        filled = fill_feature(train_df[feature])
        rates[feature] = train_df[target_column].groupby(filled).mean().to_dict()
    scores = pd.Series(0.0, index=heldout_df.index)
    for feature in features:
        scores += fill_feature(heldout_df[feature]).map(rates[feature]).fillna(global_rate)
    scores = scores / len(features)
    return (feature_score(scores, heldout_df[target_column]) + 1) / 2


def feature_report(
    config: FeatureAnalysisConfig,
    train_df: pd.DataFrame,
    heldout_df: pd.DataFrame,
    composite: pd.DataFrame,
    selected: pd.DataFrame,
    subset_metrics: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# Feature Analysis Report",
            "",
            f"- Train rows: {len(train_df)}",
            f"- Held-out rows: {len(heldout_df)}",
            f"- Selected features: {int(selected['selected'].sum())}",
            f"- Minimum prediction-power threshold: {config.min_prediction_power}",
            "",
            "## Top Composite Features",
            "",
            markdown_table(composite[["feature", "composite_score", "prediction_power"]].head(config.top_n)),
            "",
            "## Subset Performance",
            "",
            markdown_table(subset_metrics),
            "",
        ]
    )


def infer_feature_type(series: pd.Series) -> str:
    non_missing = set(series.dropna().astype("object").unique().tolist())
    if non_missing and non_missing.issubset(set(BOOLEAN_MAP)):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def feature_score(feature: pd.Series, target: pd.Series) -> float:
    y = validate_binary_target(target)
    if infer_feature_type(feature) == "numeric":
        scores = pd.to_numeric(feature, errors="coerce")
    elif infer_feature_type(feature) == "boolean":
        scores = feature.astype("object").map(BOOLEAN_MAP)
    else:
        filled = fill_feature(feature)
        rates = y.groupby(filled).mean()
        scores = filled.map(rates)
    return signed_auc(scores, y)


def signed_auc(scores: pd.Series, target: pd.Series) -> float:
    valid = scores.notna() & target.notna()
    y = target[valid].astype(int)
    x = scores[valid].astype(float)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0 or len(x) == 0:
        return math.nan
    ranks = x.rank(method="average")
    rank_sum_positive = float(ranks[y == 1].sum())
    auc = (rank_sum_positive - positives * (positives + 1) / 2) / (positives * negatives)
    return float(2 * auc - 1)


def model_matrix(
    df: pd.DataFrame, feature_columns: list[str], max_levels: int = 25
) -> pd.DataFrame:
    parts = []
    for feature in feature_columns:
        kind = infer_feature_type(df[feature])
        if kind == "numeric":
            x = pd.to_numeric(df[feature], errors="coerce")
            x = x.fillna(x.median() if x.notna().any() else 0.0)
            parts.append(standardize(x).rename(feature).to_frame())
        elif kind == "boolean":
            x = df[feature].astype("object").map(BOOLEAN_MAP).fillna(-1)
            parts.append(standardize(x).rename(feature).to_frame())
        else:
            filled = fill_feature(df[feature])
            top = filled.value_counts().head(max_levels).index
            trimmed = filled.where(filled.isin(top), "__other__")
            parts.append(pd.get_dummies(trimmed, prefix=feature, prefix_sep="__", dtype=float))
    if not parts:
        return pd.DataFrame(index=df.index)
    matrix = pd.concat(parts, axis=1)
    return matrix.loc[:, matrix.nunique(dropna=False) > 1]


def root_feature(encoded_feature: str) -> str:
    return encoded_feature.split("__", 1)[0]


def redundant_features_to_drop(composite: pd.DataFrame, redundancy_pairs: pd.DataFrame) -> set[str]:
    if redundancy_pairs.empty or composite.empty:
        return set()
    scores = composite.set_index("feature")["composite_score"].to_dict()
    drops = set()
    for _, row in redundancy_pairs.iterrows():
        a = row["feature_a"]
        b = row["feature_b"]
        if a == b:
            continue
        if scores.get(a, 0.0) >= scores.get(b, 0.0):
            drops.add(b)
        else:
            drops.add(a)
    return drops


def fill_feature(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), MISSING_TOKEN).astype(str)


def standardize(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    std = series.std()
    return (series - series.mean()) / std if std else series * 0.0


def has_two_classes(target: pd.Series) -> bool:
    return set(target.dropna().unique().tolist()) == {0, 1}


def _nan_to_zero(value: float) -> float:
    return 0.0 if pd.isna(value) else float(value)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        if value.startswith("[") and value.endswith("]"):
            values[key] = [
                item.strip().strip("'\"")
                for item in value[1:-1].split(",")
                if item.strip()
            ]
        elif value.lower() in {"true", "false"}:
            values[key] = value.lower() == "true"
        else:
            try:
                values[key] = int(value)
            except ValueError:
                try:
                    values[key] = float(value)
                except ValueError:
                    values[key] = value.strip("'\"")
    return values


def write_bar_svg(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 900
    row_h = 24
    height = max(80, 40 + row_h * len(df))
    max_value = max(df.get("composite_score", pd.Series(dtype=float)).fillna(0).tolist() + [1.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for idx, (_, row) in enumerate(df.iterrows()):
        y = 30 + idx * row_h
        value = float(row.get("composite_score", 0.0))
        bar_w = int((value / max_value) * 420) if max_value else 0
        label = svg_escape(str(row.get("feature", ""))[:54])
        lines.append(f'<text x="8" y="{y + 14}" font-size="12" font-family="sans-serif">{label}</text>')
        lines.append(f'<rect x="380" y="{y}" width="{bar_w}" height="16" fill="#2a6f97"/>')
        lines.append(f'<text x="{385 + bar_w}" y="{y + 13}" font-size="12" font-family="monospace">{value:.3f}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    safe = frame.copy()
    safe.columns = [str(column) for column in safe.columns]
    lines = [
        "| " + " | ".join(safe.columns) + " |",
        "| " + " | ".join(["---"] * len(safe.columns)) + " |",
    ]
    for _, row in safe.iterrows():
        lines.append("| " + " | ".join(markdown_cell(row[column]) for column in safe.columns) + " |")
    return "\n".join(lines)


def markdown_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|")


def svg_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def empty_ranking() -> pd.DataFrame:
    return pd.DataFrame(columns=["feature", "signed_auc", "prediction_power"])


def empty_stability() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["feature", "train_signed_auc", "heldout_signed_auc", "abs_auc_delta", "stable"]
    )


def empty_model_importance() -> pd.DataFrame:
    return pd.DataFrame(columns=["feature", "importance"])


def empty_permutation() -> pd.DataFrame:
    return pd.DataFrame(columns=["feature", "class_label", "permutation_importance"])


def empty_composite() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "feature",
            "composite_score",
            "prediction_power",
            "is_constant",
            "is_near_constant",
            "is_likely_identifier",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-safe binary feature analysis.")
    parser.add_argument("--config", help="Optional JSON/YAML config file.")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--target-column")
    parser.add_argument("--id-columns", help="Comma-separated id columns.")
    parser.add_argument("--group-columns", help="Comma-separated group columns.")
    parser.add_argument("--excluded-columns", help="Comma-separated columns to exclude.")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--min-prediction-power", type=float)
    parser.add_argument("--redundancy-threshold", type=float)
    parser.add_argument("--stability-threshold", type=float)
    parser.add_argument("--test-fraction", type=float)
    parser.add_argument("--random-state", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "target_column": args.target_column,
        "id_columns": args.id_columns,
        "group_columns": args.group_columns,
        "excluded_columns": args.excluded_columns,
        "max_rows": args.max_rows,
        "top_n": args.top_n,
        "min_prediction_power": args.min_prediction_power,
        "redundancy_threshold": args.redundancy_threshold,
        "stability_threshold": args.stability_threshold,
        "test_fraction": args.test_fraction,
        "random_state": args.random_state,
    }
    config = merge_config(args.config, overrides)
    result = run_feature_analysis(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
