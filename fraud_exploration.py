"""Compact workpacket-2 exploration for IEEE-CIS fraud data.

This module keeps the exploration path intentionally small and local to one
file. It covers fraud-only fact finding, dataset audit, target association,
missingness signal, numeric correlations, simple coefficient/LDA-style
direction summaries, and sampled pairwise similarity over fraud cases.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd

from fraud_preprocessing import FraudDataLoader


SECONDS_PER_DAY = 86400
MISSING_TOKEN = "__missing__"
BOOLEAN_LIKE_VALUES = {"T", "F", True, False, 0, 1}


def validate_binary_target(
    target: pd.Series, positive_label=1, negative_label=0
) -> pd.Series:
    """Validate and encode a two-class target as integer 0/1."""
    unique = set(target.dropna().unique().tolist())
    expected = {negative_label, positive_label}
    if unique != expected:
        raise ValueError(
            f"Expected target labels {sorted(expected)!r}; found {sorted(unique)!r}"
        )
    return target.map({negative_label: 0, positive_label: 1}).astype(int)


def infer_feature_type(series: pd.Series) -> str:
    """Classify a feature as boolean, numeric, or categorical."""
    non_missing = set(series.dropna().astype("object").unique().tolist())
    if non_missing and non_missing.issubset(BOOLEAN_LIKE_VALUES):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


class FraudFactFinder:
    """Descriptive analysis for confirmed fraud rows."""

    def __init__(self, fraud_df: pd.DataFrame) -> None:
        self.df = fraud_df.copy()

    def summary(self) -> dict:
        if self.df.empty:
            return {
                "n_fraud_cases": 0,
                "transaction_dt_days_min": math.nan,
                "transaction_dt_days_max": math.nan,
                "transaction_dt_days_span": math.nan,
                "transaction_amt_total": 0.0,
                "transaction_amt_mean": math.nan,
                "transaction_amt_median": math.nan,
            }

        dt_days = pd.to_numeric(self.df["TransactionDT"], errors="coerce") / SECONDS_PER_DAY
        amount = pd.to_numeric(self.df["TransactionAmt"], errors="coerce")
        return {
            "n_fraud_cases": int(len(self.df)),
            "transaction_dt_days_min": float(dt_days.min()),
            "transaction_dt_days_max": float(dt_days.max()),
            "transaction_dt_days_span": float(dt_days.max() - dt_days.min()),
            "transaction_amt_total": float(amount.sum()),
            "transaction_amt_mean": float(amount.mean()),
            "transaction_amt_median": float(amount.median()),
        }

    def distribution(self, column: str, top_n: int = 10, normalize: bool = True) -> pd.Series:
        if column not in self.df:
            return pd.Series(dtype=float if normalize else int)
        return self.df[column].value_counts(normalize=normalize, dropna=False).head(top_n)

    def numeric_profile(self, column: str) -> dict:
        series = pd.to_numeric(self.df[column], errors="coerce") if column in self.df else pd.Series(dtype=float)
        non_missing = series.dropna()
        quantiles = non_missing.quantile([0.05, 0.25, 0.75, 0.95]) if len(non_missing) else {}
        return {
            "count": int(len(non_missing)),
            "missing_count": int(series.isna().sum()),
            "mean": _float_or_nan(non_missing.mean()),
            "median": _float_or_nan(non_missing.median()),
            "std": _float_or_nan(non_missing.std()),
            "min": _float_or_nan(non_missing.min()),
            "max": _float_or_nan(non_missing.max()),
            "q05": _float_or_nan(quantiles.get(0.05, math.nan)),
            "q25": _float_or_nan(quantiles.get(0.25, math.nan)),
            "q75": _float_or_nan(quantiles.get(0.75, math.nan)),
            "q95": _float_or_nan(quantiles.get(0.95, math.nan)),
        }

    def time_bucket_summary(self, bins: int = 24) -> pd.DataFrame:
        if self.df.empty or "TransactionDT" not in self.df:
            return pd.DataFrame(columns=["bucket", "count", "pct_of_fraud"])
        seconds = pd.to_numeric(self.df["TransactionDT"], errors="coerce") % SECONDS_PER_DAY
        bucket = (seconds / (SECONDS_PER_DAY / bins)).fillna(-1).astype(int).clip(0, bins - 1)
        counts = bucket.value_counts(sort=False).reindex(range(bins), fill_value=0)
        return pd.DataFrame(
            {
                "bucket": counts.index,
                "count": counts.values,
                "pct_of_fraud": counts.values / len(self.df),
            }
        )

    def common_combinations(self, columns: list[str], top_n: int = 15) -> pd.DataFrame:
        present = [col for col in columns if col in self.df]
        if not present or self.df.empty:
            return pd.DataFrame(columns=present + ["count", "pct_of_fraud"])
        grouped = (
            self.df.groupby(present, dropna=False, observed=True)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        grouped["pct_of_fraud"] = grouped["count"] / len(self.df)
        return grouped

    def email_domain_summary(self, top_n: int = 10) -> dict:
        if self.df.empty or not {"P_emaildomain", "R_emaildomain"}.issubset(self.df.columns):
            return {
                "p_emaildomain_top": pd.Series(dtype=float),
                "r_emaildomain_top": pd.Series(dtype=float),
                "match_pct": math.nan,
                "differ_pct": math.nan,
                "either_missing_pct": math.nan,
            }
        p = self.df["P_emaildomain"]
        r = self.df["R_emaildomain"]
        either_missing = p.isna() | r.isna()
        both_present = ~either_missing
        match = both_present & (p.astype("object") == r.astype("object"))
        differ = both_present & ~match
        total = len(self.df)
        return {
            "p_emaildomain_top": self.distribution("P_emaildomain", top_n=top_n),
            "r_emaildomain_top": self.distribution("R_emaildomain", top_n=top_n),
            "match_pct": float(match.sum() / total),
            "differ_pct": float(differ.sum() / total),
            "either_missing_pct": float(either_missing.sum() / total),
        }

    def card_type_summary(self) -> pd.DataFrame:
        if self.df.empty or not {"card4", "card6"}.issubset(self.df.columns):
            return pd.DataFrame()
        return pd.crosstab(self.df["card4"], self.df["card6"], dropna=False, margins=True)

    def to_markdown(self) -> str:
        lines = ["# Fraud Fact Finder Report", ""]
        lines.extend(_dict_markdown("Summary", self.summary()))
        lines.extend(_series_markdown("ProductCD distribution", self.distribution("ProductCD")))
        lines.extend(_series_markdown("DeviceType distribution", self.distribution("DeviceType")))
        lines.extend(_frame_markdown("Time buckets", self.time_bucket_summary()))
        lines.extend(_frame_markdown("Common product/device combinations", self.common_combinations(["ProductCD", "DeviceType"])))
        lines.extend(_frame_markdown("Common email/card combinations", self.common_combinations(["P_emaildomain", "R_emaildomain", "card4", "card6"])))
        lines.extend(_frame_markdown("Card type summary", self.card_type_summary()))
        email = self.email_domain_summary()
        lines.extend(
            _dict_markdown(
                "Email domain match rates",
                {
                    "match_pct": email["match_pct"],
                    "differ_pct": email["differ_pct"],
                    "either_missing_pct": email["either_missing_pct"],
                },
            )
        )
        lines.extend(_series_markdown("P_emaildomain top values", email["p_emaildomain_top"]))
        lines.extend(_series_markdown("R_emaildomain top values", email["r_emaildomain_top"]))
        lines.extend(_dict_markdown("TransactionAmt profile", self.numeric_profile("TransactionAmt")))
        return "\n".join(lines)

    def to_report(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown())


def audit_dataset(
    df: pd.DataFrame,
    target_column: str = "isFraud",
    id_columns: Iterable[str] = ("TransactionID",),
    near_zero_variance_threshold: float = 0.01,
    identifier_uniqueness_threshold: float = 0.95,
) -> pd.DataFrame:
    """Return one row per feature with missingness, variance, and quality flags."""
    y = df[target_column] if target_column in df else pd.Series(index=df.index, dtype=float)
    features = [c for c in df.columns if c != target_column and c not in set(id_columns)]
    duplicate_map = _find_duplicate_columns(df[features])
    rows = []

    for col in features:
        series = df[col]
        feature_type = infer_feature_type(series)
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        if feature_type in {"numeric", "boolean"}:
            numeric = _numeric_series(series)
            variance = _float_or_nan(numeric.dropna().var())
            non_missing_numeric = numeric.dropna()
            n_invalid = int(non_missing_numeric.map(math.isinf).sum()) if len(non_missing_numeric) else 0
            frequencies = numeric.value_counts(normalize=True, dropna=True)
        else:
            variance = math.nan
            n_invalid = 0
            frequencies = series.value_counts(normalize=True, dropna=True)
        most_frequent_fraction = _float_or_nan(frequencies.iloc[0]) if len(frequencies) else math.nan
        looks_discrete = feature_type == "categorical" or pd.api.types.is_integer_dtype(series)

        rows.append(
            {
                "feature": col,
                "feature_type": feature_type,
                "n_missing": n_missing,
                "missing_fraction": n_missing / len(df) if len(df) else math.nan,
                "n_unique": n_unique,
                "variance": variance,
                "most_frequent_fraction": most_frequent_fraction,
                "is_constant": n_unique <= 1,
                "is_near_zero_variance": bool(
                    (pd.notna(variance) and variance <= near_zero_variance_threshold)
                    or (pd.notna(most_frequent_fraction) and most_frequent_fraction >= 0.999)
                ),
                "is_likely_identifier": bool(
                    len(df) and looks_discrete and (n_unique / len(df)) >= identifier_uniqueness_threshold
                ),
                "is_duplicate": col in duplicate_map,
                "duplicate_of": duplicate_map.get(col),
                "missingness_class_0": _class_missingness(series, y, 0),
                "missingness_class_1": _class_missingness(series, y, 1),
                "n_invalid_or_infinite": n_invalid,
            }
        )

    return pd.DataFrame(rows)


def analyze_univariate_features(
    df: pd.DataFrame,
    target_column: str = "isFraud",
    id_columns: Iterable[str] = ("TransactionID",),
) -> pd.DataFrame:
    """Compute simple per-feature association with the binary target."""
    y = validate_binary_target(df[target_column])
    rows = []
    for col in [c for c in df.columns if c != target_column and c not in set(id_columns)]:
        x = df[col]
        feature_type = infer_feature_type(x)
        row = {"feature": col, "feature_type": feature_type}
        if feature_type == "numeric":
            numeric = _numeric_series(x)
            row.update(_numeric_association(numeric, y))
        else:
            row.update(_categorical_association(x, y))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("association_strength", ascending=False).reset_index(drop=True)


def analyze_missingness_signal(
    df: pd.DataFrame,
    target_column: str = "isFraud",
    id_columns: Iterable[str] = ("TransactionID",),
) -> pd.DataFrame:
    """Measure whether feature missingness differs by target class."""
    y = validate_binary_target(df[target_column])
    rows = []
    for col in [c for c in df.columns if c != target_column and c not in set(id_columns)]:
        missing = df[col].isna()
        if not missing.any():
            continue
        assoc = _categorical_association(missing, y)
        rows.append(
            {
                "feature": col,
                "missing_fraction": float(missing.mean()),
                "missing_rate_class_0": float(missing[y == 0].mean()),
                "missing_rate_class_1": float(missing[y == 1].mean()),
                "missing_rate_diff_class_1_minus_0": float(missing[y == 1].mean() - missing[y == 0].mean()),
                "missingness_auc_strength": assoc["association_strength"],
            }
        )
    return pd.DataFrame(rows).sort_values("missingness_auc_strength", ascending=False).reset_index(drop=True)


def numeric_feature_correlations(
    df: pd.DataFrame,
    target_column: str = "isFraud",
    id_columns: Iterable[str] = ("TransactionID",),
    threshold: float = 0.8,
) -> pd.DataFrame:
    """Report numeric feature pairs whose Pearson correlation exceeds threshold."""
    excluded = set(id_columns) | {target_column}
    numeric = df[[c for c in df.columns if c not in excluded]].select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "correlation"])
    corr = numeric.corr()
    rows = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            value = corr.loc[a, b]
            if pd.notna(value) and abs(value) >= threshold:
                rows.append({"feature_a": a, "feature_b": b, "correlation": float(value)})
    return pd.DataFrame(rows).sort_values("correlation", key=lambda s: s.abs(), ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["feature_a", "feature_b", "correlation"])


def standardized_logistic_coefficients(
    df: pd.DataFrame,
    target_column: str = "isFraud",
    id_columns: Iterable[str] = ("TransactionID",),
    top_n: int = 50,
) -> pd.DataFrame:
    """Fit standardized logistic regression and report coefficient directions."""
    y = validate_binary_target(df[target_column])
    excluded = set(id_columns) | {target_column}
    try:
        from sklearn.linear_model import LogisticRegression

        X = _model_matrix(df[[c for c in df.columns if c not in excluded]])
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
        model.fit(X, y)
        rows = [
            {
                "feature": feature,
                "level": None,
                "coefficient": float(coef),
                "abs_coefficient": abs(float(coef)),
                "favored_class": 1 if coef > 0 else 0,
            }
            for feature, coef in zip(X.columns, model.coef_.ravel())
        ]
        return pd.DataFrame(rows).sort_values("abs_coefficient", ascending=False).head(top_n).reset_index(drop=True)
    except Exception:
        pass

    rows = []
    for col in [c for c in df.columns if c not in excluded]:
        x = df[col]
        feature_type = infer_feature_type(x)
        if feature_type == "numeric":
            z = _standardize(_numeric_series(x))
            coef = _safe_corr(z, y)
            rows.append({"feature": col, "level": None, "coefficient": coef, "abs_coefficient": abs(coef), "favored_class": 1 if coef > 0 else 0})
        else:
            filled = _filled_category(x)
            for level in filled.value_counts().head(12).index:
                indicator = (filled == level).astype(int)
                coef = _safe_corr(indicator, y)
                rows.append({"feature": col, "level": str(level), "coefficient": coef, "abs_coefficient": abs(coef), "favored_class": 1 if coef > 0 else 0})
    return pd.DataFrame(rows).sort_values("abs_coefficient", ascending=False).head(top_n).reset_index(drop=True)


def lda_separability_summary(
    df: pd.DataFrame,
    target_column: str = "isFraud",
    id_columns: Iterable[str] = ("TransactionID",),
    top_n: int = 50,
) -> pd.DataFrame:
    """Fit one-component LDA and report class-separation loadings."""
    y = validate_binary_target(df[target_column])
    excluded = set(id_columns) | {target_column}
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        X = _model_matrix(df[[c for c in df.columns if c not in excluded]])
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(X, y)
        rows = [
            {
                "feature": feature,
                "lda_loading": float(loading),
                "abs_lda_loading": abs(float(loading)),
            }
            for feature, loading in zip(X.columns, lda.scalings_.ravel())
        ]
        return pd.DataFrame(rows).sort_values("abs_lda_loading", ascending=False).head(top_n).reset_index(drop=True)
    except Exception:
        pass

    rows = []
    for col in [c for c in df.columns if c not in excluded]:
        if infer_feature_type(df[col]) != "numeric":
            continue
        x = _numeric_series(df[col])
        x0 = x[y == 0].dropna()
        x1 = x[y == 1].dropna()
        pooled = math.sqrt((_float_or_zero(x0.var()) + _float_or_zero(x1.var())) / 2)
        loading = float((x1.mean() - x0.mean()) / pooled) if pooled else math.nan
        rows.append({"feature": col, "lda_loading": loading, "abs_lda_loading": abs(loading) if pd.notna(loading) else math.nan})
    return pd.DataFrame(rows).sort_values("abs_lda_loading", ascending=False).head(top_n).reset_index(drop=True)


def shap_status() -> dict:
    """Optional SHAP hook; intentionally does not require shap for this repo."""
    if importlib.util.find_spec("shap") is None:
        return {"available": False, "reason": "shap is not installed"}
    return {"available": True, "reason": "shap is installed; add a fitted model to compute attributions"}


def sample_fraud_cases_for_similarity(
    fraud_df: pd.DataFrame,
    sample_size: int = 50,
    cluster_column: str = "ProductCD",
    random_state: int = 42,
) -> pd.DataFrame:
    """Cluster-first sample using a stable categorical proxy when full clustering is overkill."""
    if fraud_df.empty:
        return fraud_df.copy()
    sample_size = min(sample_size, len(fraud_df))
    if cluster_column not in fraud_df:
        return fraud_df.sample(n=sample_size, random_state=random_state)
    parts = []
    counts = fraud_df[cluster_column].fillna(MISSING_TOKEN).value_counts()
    allocation = (sample_size * counts / counts.sum()).apply(math.floor).astype(int)
    while allocation.sum() < sample_size:
        for key in counts.index:
            if allocation.sum() >= sample_size:
                break
            if allocation[key] < counts[key]:
                allocation[key] += 1
    for key, n in allocation.items():
        if n <= 0:
            continue
        mask = fraud_df[cluster_column].fillna(MISSING_TOKEN) == key
        parts.append(fraud_df[mask].sample(n=int(n), random_state=random_state))
    return pd.concat(parts).sort_index() if parts else fraud_df.head(0).copy()


def pairwise_similarity(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    numeric_distance_percentile: float = 0.9,
) -> pd.DataFrame:
    """Compute pairwise fraud-case similarity, excluding missing comparison units."""
    if columns is None:
        columns = _default_similarity_columns(df)
    columns = [col for col in columns if col in df.columns]
    labels = df["TransactionID"].astype(str).tolist() if "TransactionID" in df else [str(i) for i in range(len(df))]
    matrix = []
    numeric_cutoffs = _numeric_distance_cutoffs(df, columns, numeric_distance_percentile)
    for i in range(len(df)):
        row = []
        for j in range(len(df)):
            scores = []
            for col in columns:
                a = df.iloc[i][col]
                b = df.iloc[j][col]
                if pd.isna(a) or pd.isna(b):
                    continue
                kind = infer_feature_type(df[col])
                if kind == "numeric":
                    cutoff = numeric_cutoffs.get(col, 0.0)
                    diff = abs(float(a) - float(b))
                    score = 1.0 if cutoff == 0 else max(0.0, 1.0 - diff / cutoff)
                else:
                    score = 1.0 if a == b else 0.0
                scores.append(score)
            row.append(float(sum(scores) / len(scores)) if scores else math.nan)
        matrix.append(row)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def similarity_summary(matrix: pd.DataFrame) -> dict:
    values = []
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            value = matrix.iloc[i, j]
            if pd.notna(value):
                values.append(float(value))
    if not values:
        return {"n_sampled": len(matrix), "mean_similarity": math.nan, "median_similarity": math.nan, "n_nan_pairs": int(matrix.isna().sum().sum())}
    s = pd.Series(values)
    return {
        "n_sampled": len(matrix),
        "mean_similarity": float(s.mean()),
        "median_similarity": float(s.median()),
        "min_similarity": float(s.min()),
        "max_similarity": float(s.max()),
        "n_nan_pairs": int(matrix.isna().sum().sum()),
    }


def run_exploration(
    data_dir: str | Path = "data/example_subset",
    output_dir: str | Path = "outputs/example_exploration",
    similarity_sample_size: int = 30,
) -> dict:
    """Run the compact workpacket-2 workflow and write outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = FraudDataLoader(data_dir).load_train()
    fraud = train[train["isFraud"] == 1].copy()

    fact_finder = FraudFactFinder(fraud)
    fact_finder.to_report(output_dir / "fraud_fact_finder.md")

    audit = audit_dataset(train)
    audit.to_csv(output_dir / "dataset_audit.csv", index=False)

    result = {
        "rows": int(len(train)),
        "fraud_rows": int(len(fraud)),
        "outputs": ["fraud_fact_finder.md", "dataset_audit.csv"],
        "shap": shap_status(),
    }

    if set(train["isFraud"].dropna().unique()) == {0, 1}:
        univariate = analyze_univariate_features(train)
        missingness = analyze_missingness_signal(train)
        correlations = numeric_feature_correlations(train)
        coefficients = standardized_logistic_coefficients(train)
        lda = lda_separability_summary(train)
        univariate.to_csv(output_dir / "univariate_feature_analysis.csv", index=False)
        missingness.to_csv(output_dir / "missingness_signal.csv", index=False)
        correlations.to_csv(output_dir / "numeric_correlations.csv", index=False)
        coefficients.to_csv(output_dir / "standardized_coefficients.csv", index=False)
        lda.to_csv(output_dir / "lda_separability.csv", index=False)
        write_bar_svg(univariate.head(20), "feature", "association_strength", output_dir / "univariate_strength.svg")
        write_bar_svg(missingness.head(20), "feature", "missingness_auc_strength", output_dir / "missingness_signal.svg")
        result["outputs"].extend(
            [
                "univariate_feature_analysis.csv",
                "missingness_signal.csv",
                "numeric_correlations.csv",
                "standardized_coefficients.csv",
                "lda_separability.csv",
                "univariate_strength.svg",
                "missingness_signal.svg",
            ]
        )
    else:
        (output_dir / "binary_feature_analysis_skipped.txt").write_text(
            "Binary feature analysis requires both isFraud classes. "
            f"Found labels: {sorted(train['isFraud'].dropna().unique().tolist())}\n"
        )
        result["outputs"].append("binary_feature_analysis_skipped.txt")

    if len(fraud) >= 2:
        sampled = sample_fraud_cases_for_similarity(fraud, sample_size=similarity_sample_size)
        matrix = pairwise_similarity(sampled)
        matrix.to_csv(output_dir / "fraud_similarity_matrix.csv")
        matrix.to_pickle(output_dir / "fraud_similarity_matrix.pkl")
        summary = similarity_summary(matrix)
        (output_dir / "fraud_similarity_summary.json").write_text(json.dumps(summary, indent=2))
        write_heatmap_svg(matrix, output_dir / "fraud_similarity_heatmap.svg")
        result["similarity"] = summary
        result["outputs"].extend(
            [
                "fraud_similarity_matrix.csv",
                "fraud_similarity_matrix.pkl",
                "fraud_similarity_summary.json",
                "fraud_similarity_heatmap.svg",
            ]
        )
    else:
        (output_dir / "fraud_similarity_skipped.txt").write_text(
            f"Fraud similarity requires at least 2 fraud rows. Found {len(fraud)}.\n"
        )
        result["outputs"].append("fraud_similarity_skipped.txt")

    (output_dir / "run_summary.json").write_text(json.dumps(result, indent=2))
    result["outputs"].append("run_summary.json")
    return result


def write_bar_svg(df: pd.DataFrame, label_col: str, value_col: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 900
    row_h = 24
    height = max(80, 40 + row_h * len(df))
    max_value = max([abs(float(v)) for v in df[value_col].fillna(0).tolist()] + [1.0])
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>']
    for idx, (_, row) in enumerate(df.iterrows()):
        y = 30 + idx * row_h
        value = float(row[value_col]) if pd.notna(row[value_col]) else 0.0
        bar_w = int((abs(value) / max_value) * 420)
        color = "#1f77b4" if value >= 0 else "#d62728"
        label = _svg_escape(str(row[label_col])[:48])
        lines.append(f'<text x="8" y="{y + 14}" font-size="12" font-family="sans-serif">{label}</text>')
        lines.append(f'<rect x="360" y="{y}" width="{bar_w}" height="16" fill="{color}"/>')
        lines.append(f'<text x="{365 + bar_w}" y="{y + 13}" font-size="12" font-family="monospace">{value:.3f}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def write_heatmap_svg(matrix: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(matrix)
    cell = max(4, min(24, 420 // max(n, 1)))
    size = max(80, n * cell + 20)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">', '<rect width="100%" height="100%" fill="white"/>']
    for i in range(n):
        for j in range(n):
            value = matrix.iloc[i, j]
            shade = 240 if pd.isna(value) else int(240 - 180 * max(0.0, min(1.0, float(value))))
            lines.append(f'<rect x="{10 + j * cell}" y="{10 + i * cell}" width="{cell}" height="{cell}" fill="rgb({shade},{shade},255)"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def _find_duplicate_columns(df: pd.DataFrame) -> dict[str, str]:
    duplicate_of = {}
    representatives = []
    for col in df.columns:
        match = next((rep for rep in representatives if df[col].equals(df[rep])), None)
        if match is None:
            representatives.append(col)
        else:
            duplicate_of[col] = match
    return duplicate_of


def _numeric_series(series: pd.Series) -> pd.Series:
    if infer_feature_type(series) == "boolean":
        return series.astype("object").map({"T": 1, "F": 0, True: 1, False: 0, 1: 1, 0: 0})
    return pd.to_numeric(series, errors="coerce")


def _class_missingness(series: pd.Series, y: pd.Series, cls: int) -> float:
    if y.empty or cls not in set(y.dropna().unique()):
        return math.nan
    mask = y == cls
    return float(series[mask].isna().mean()) if mask.any() else math.nan


def _numeric_association(x: pd.Series, y: pd.Series) -> dict:
    valid = x.notna()
    class0 = x[valid & (y == 0)]
    class1 = x[valid & (y == 1)]
    corr = _safe_corr(x[valid], y[valid])
    return {
        "mean_class_0": _float_or_nan(class0.mean()),
        "mean_class_1": _float_or_nan(class1.mean()),
        "median_class_0": _float_or_nan(class0.median()),
        "median_class_1": _float_or_nan(class1.median()),
        "signed_mean_difference": _float_or_nan(class1.mean() - class0.mean()),
        "point_biserial_correlation": corr,
        "signed_auc": _signed_auc_from_scores(x, y),
        "association_strength": abs(_signed_auc_from_scores(x, y)),
    }


def _categorical_association(x: pd.Series, y: pd.Series) -> dict:
    filled = _filled_category(x)
    levels = filled.value_counts(normalize=True)
    rates = y.groupby(filled).mean()
    global_rate = float(y.mean())
    scores = filled.map(rates).fillna(global_rate)
    signed_auc = _signed_auc_from_scores(scores, y)
    class_rates = pd.DataFrame({"x": filled, "y": y}).groupby("x")["y"].mean()
    return {
        "n_levels": int(len(levels)),
        "most_frequent_level": str(levels.index[0]) if len(levels) else None,
        "most_frequent_level_frequency": _float_or_nan(levels.iloc[0]) if len(levels) else math.nan,
        "max_target_rate": _float_or_nan(class_rates.max()),
        "min_target_rate": _float_or_nan(class_rates.min()),
        "target_rate_range": _float_or_nan(class_rates.max() - class_rates.min()),
        "signed_auc": signed_auc,
        "association_strength": abs(signed_auc),
    }


def _signed_auc_from_scores(scores: pd.Series, y: pd.Series) -> float:
    valid = scores.notna() & y.notna()
    yv = y[valid].astype(int)
    xv = scores[valid].astype(float)
    n_pos = int((yv == 1).sum())
    n_neg = int((yv == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = xv.rank(method="average")
    rank_sum_pos = float(ranks[yv == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(2 * auc - 1)


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    joined = pd.concat([x, y], axis=1).dropna()
    if len(joined) < 3 or joined.iloc[:, 0].nunique() < 2 or joined.iloc[:, 1].nunique() < 2:
        return math.nan
    return _float_or_nan(joined.iloc[:, 0].astype(float).corr(joined.iloc[:, 1].astype(float)))


def _standardize(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    std = x.std()
    return (x - x.mean()) / std if std else x * 0


def _filled_category(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), MISSING_TOKEN).astype(str)


def _numeric_distance_cutoffs(df: pd.DataFrame, columns: list[str], percentile: float) -> dict[str, float]:
    cutoffs = {}
    for col in columns:
        if infer_feature_type(df[col]) != "numeric":
            continue
        values = _numeric_series(df[col]).dropna().tolist()
        diffs = [abs(a - b) for idx, a in enumerate(values) for b in values[idx + 1 :]]
        cutoffs[col] = float(pd.Series(diffs).quantile(percentile)) if diffs else 0.0
    return cutoffs


def _default_similarity_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "TransactionAmt",
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "DeviceInfo",
        "M1",
        "M2",
        "M3",
    ]
    present = [col for col in preferred if col in df.columns]
    if present:
        return present
    return [col for col in df.columns if col not in {"TransactionID", "isFraud"}][:20]


def _model_matrix(df: pd.DataFrame, max_levels: int = 25) -> pd.DataFrame:
    """Build a small standardized numeric/one-hot design matrix."""
    parts = []
    for col in df.columns:
        kind = infer_feature_type(df[col])
        if kind == "numeric":
            x = _numeric_series(df[col])
            x = x.fillna(x.median() if x.notna().any() else 0.0)
            parts.append(_standardize(x).rename(col).to_frame())
        elif kind == "boolean":
            x = _numeric_series(df[col]).fillna(-1)
            parts.append(_standardize(x).rename(col).to_frame())
        else:
            filled = _filled_category(df[col])
            top = filled.value_counts().head(max_levels).index
            trimmed = filled.where(filled.isin(top), "__other__")
            dummies = pd.get_dummies(trimmed, prefix=col, prefix_sep="__", dtype=float)
            parts.append(dummies)
    if not parts:
        return pd.DataFrame(index=df.index)
    matrix = pd.concat(parts, axis=1)
    return matrix.loc[:, matrix.nunique(dropna=False) > 1]


def _float_or_nan(value) -> float:
    return float(value) if pd.notna(value) else math.nan


def _float_or_zero(value) -> float:
    return float(value) if pd.notna(value) else 0.0


def _dict_markdown(title: str, values: dict) -> list[str]:
    lines = [f"## {title}", ""]
    for key, value in values.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    return lines


def _series_markdown(title: str, series: pd.Series) -> list[str]:
    lines = [f"## {title}", ""]
    if len(series):
        lines.append(_markdown_table(series.rename("value").to_frame().reset_index()))
    else:
        lines.append("_No rows._")
    lines.append("")
    return lines


def _frame_markdown(title: str, frame: pd.DataFrame) -> list[str]:
    lines = [f"## {title}", ""]
    if len(frame):
        lines.append(_markdown_table(frame))
    else:
        lines.append("_No rows._")
    lines.append("")
    return lines


def _svg_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    safe = frame.copy()
    safe.columns = [str(col) for col in safe.columns]
    rows = [
        "| " + " | ".join(safe.columns) + " |",
        "| " + " | ".join(["---"] * len(safe.columns)) + " |",
    ]
    for _, row in safe.iterrows():
        values = [_markdown_cell(row[col]) for col in safe.columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _markdown_cell(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compact fraud exploration.")
    parser.add_argument("--data-dir", default="data/example_subset")
    parser.add_argument("--output-dir", default="outputs/example_exploration")
    parser.add_argument("--similarity-sample-size", type=int, default=30)
    args = parser.parse_args()

    result = run_exploration(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        similarity_sample_size=args.similarity_sample_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
