"""Create lightweight SVG plots from fraud workflow outputs.

This module intentionally avoids a plotting dependency. It reads CSV and JSON
artifacts from exploration, feature analysis, and model training, then writes
simple SVG charts plus an index page.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def run_results_plots(
    exploration_dir: str | Path = "outputs/full_exploration",
    feature_dir: str | Path = "outputs/full_feature_analysis",
    model_dir: str | Path = "outputs/full_model_training",
    output_dir: str | Path = "outputs/full_results_plots",
    top_n: int = 20,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []

    metrics = read_csv(Path(model_dir) / "model_comparison_metrics.csv")
    if not metrics.empty:
        path = output / "model_metric_comparison.svg"
        write_grouped_metric_bars(
            metrics,
            ["roc_auc", "pr_auc", "precision", "recall", "f1"],
            path,
            "Model Performance Metrics",
        )
        artifacts.append(path.name)

    confusion = read_csv(Path(model_dir) / "confusion_matrices.csv")
    if not confusion.empty:
        path = output / "confusion_matrices.svg"
        write_confusion_grid(confusion, path)
        artifacts.append(path.name)

    tree_rules = read_csv(Path(model_dir) / "decision_tree_rules.csv")
    if not tree_rules.empty and "prediction" in tree_rules:
        path = output / "decision_tree_leaf_predictions.svg"
        write_horizontal_bars(
            tree_rules.assign(leaf=[f"leaf_{i + 1}" for i in range(len(tree_rules))]),
            label_column="leaf",
            value_column="prediction",
            path=path,
            title="Decision Tree Leaf Fraud Probabilities",
            top_n=top_n,
        )
        artifacts.append(path.name)

    selected = read_csv(Path(feature_dir) / "selected_features.csv")
    if not selected.empty and "selected" in selected:
        counts = (
            selected.assign(status=selected["selected"].map({True: "selected", False: "dropped"}).fillna(selected["selected"].astype(str)))
            .groupby("status")
            .size()
            .reset_index(name="count")
        )
        path = output / "feature_selection_counts.svg"
        write_horizontal_bars(
            counts,
            label_column="status",
            value_column="count",
            path=path,
            title="Feature Selection Counts",
            top_n=top_n,
        )
        artifacts.append(path.name)

    composite = read_csv(Path(feature_dir) / "composite_feature_ranking.csv")
    if not composite.empty and {"feature", "composite_score"}.issubset(composite.columns):
        path = output / "top_composite_features.svg"
        write_horizontal_bars(
            composite,
            label_column="feature",
            value_column="composite_score",
            path=path,
            title="Top Composite Feature Scores",
            top_n=top_n,
        )
        artifacts.append(path.name)

    audit = read_csv(Path(exploration_dir) / "dataset_audit.csv")
    if not audit.empty and {"feature", "missing_fraction"}.issubset(audit.columns):
        missing = audit.sort_values("missing_fraction", ascending=False)
        path = output / "top_missing_features.svg"
        write_horizontal_bars(
            missing,
            label_column="feature",
            value_column="missing_fraction",
            path=path,
            title="Features With Most Missing Values",
            top_n=top_n,
        )
        artifacts.append(path.name)

    summary = {
        "exploration_dir": str(exploration_dir),
        "feature_dir": str(feature_dir),
        "model_dir": str(model_dir),
        "plots_written": artifacts,
    }
    (output / "plot_summary.json").write_text(json.dumps(summary, indent=2))
    (output / "plot_index.md").write_text(plot_index_markdown(artifacts))
    return summary


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_grouped_metric_bars(
    frame: pd.DataFrame,
    metric_columns: list[str],
    path: Path,
    title: str,
) -> None:
    width = 1100
    row_h = 34
    left = 170
    height = 90 + row_h * len(frame) * len(metric_columns)
    lines = svg_header(width, height, title)
    y = 58
    colors = ["#2a6f97", "#7a9e3f", "#c26a2e", "#6a4c93", "#8f2d56"]
    for _, row in frame.iterrows():
        model = str(row.get("model", "model"))
        lines.append(text(12, y + 14, model, size=13, weight="bold"))
        for idx, metric in enumerate(metric_columns):
            value = numeric(row.get(metric, 0.0))
            bar_w = max(0, min(1, value)) * 520
            yy = y + 22 + idx * row_h
            lines.append(text(left, yy + 13, metric, size=12))
            lines.append(rect(left + 110, yy, bar_w, 16, colors[idx % len(colors)]))
            lines.append(text(left + 120 + bar_w, yy + 13, f"{value:.3f}", size=12))
        y += row_h * len(metric_columns) + 12
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def write_confusion_grid(frame: pd.DataFrame, path: Path) -> None:
    width = 900
    cell = 72
    row_h = 124
    height = 80 + row_h * len(frame)
    lines = svg_header(width, height, "Confusion Matrices By Model")
    y = 58
    for _, row in frame.iterrows():
        model = str(row.get("model", "model"))
        values = {
            "TN": int(row.get("tn", 0)),
            "FP": int(row.get("fp", 0)),
            "FN": int(row.get("fn", 0)),
            "TP": int(row.get("tp", 0)),
        }
        max_value = max(values.values()) or 1
        lines.append(text(12, y + 18, model, size=14, weight="bold"))
        positions = [("TN", 180, y), ("FP", 260, y), ("FN", 180, y + 44), ("TP", 260, y + 44)]
        for label, x, yy in positions:
            intensity = values[label] / max_value
            color = shade("#2a6f97", intensity)
            lines.append(rect(x, yy, cell, 34, color))
            lines.append(text(x + 7, yy + 15, label, size=12, color="white"))
            lines.append(text(x + 7, yy + 29, str(values[label]), size=12, color="white"))
        y += row_h
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def write_horizontal_bars(
    frame: pd.DataFrame,
    label_column: str,
    value_column: str,
    path: Path,
    title: str,
    top_n: int = 20,
) -> None:
    data = frame[[label_column, value_column]].copy()
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce").fillna(0.0)
    data = data.sort_values(value_column, ascending=False).head(top_n)
    width = 1000
    row_h = 28
    height = max(100, 58 + row_h * len(data))
    left = 330
    max_value = float(data[value_column].max()) if len(data) else 1.0
    max_value = max(max_value, 1e-12)
    lines = svg_header(width, height, title)
    for idx, (_, row) in enumerate(data.iterrows()):
        y = 54 + idx * row_h
        label = str(row[label_column])[:52]
        value = float(row[value_column])
        bar_w = (value / max_value) * 520
        lines.append(text(12, y + 13, label, size=12))
        lines.append(rect(left, y, bar_w, 16, "#2a6f97"))
        lines.append(text(left + bar_w + 8, y + 13, f"{value:.4g}", size=12))
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def plot_index_markdown(artifacts: list[str]) -> str:
    lines = ["# Fraud Results Plot Index", ""]
    if not artifacts:
        lines.append("No plot artifacts were written because no expected input files were found.")
    for artifact in artifacts:
        lines.append(f"- [{artifact}]({artifact})")
    lines.append("")
    return "\n".join(lines)


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(12, 28, title, size=20, weight="bold"),
    ]


def rect(x: float, y: float, width: float, height: float, fill: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{fill}"/>'


def text(
    x: float,
    y: float,
    value: str,
    size: int = 12,
    color: str = "#222",
    weight: str = "normal",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="sans-serif" font-weight="{weight}" fill="{color}">{escape(value)}</text>'
    )


def shade(base: str, intensity: float) -> str:
    intensity = max(0.15, min(1.0, intensity))
    r = int(int(base[1:3], 16) * intensity + 255 * (1 - intensity))
    g = int(int(base[3:5], 16) * intensity + 255 * (1 - intensity))
    b = int(int(base[5:7], 16) * intensity + 255 * (1 - intensity))
    return f"#{r:02x}{g:02x}{b:02x}"


def numeric(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot fraud workflow output artifacts.")
    parser.add_argument("--exploration-dir", default="outputs/full_exploration")
    parser.add_argument("--feature-dir", default="outputs/full_feature_analysis")
    parser.add_argument("--model-dir", default="outputs/full_model_training")
    parser.add_argument("--output-dir", default="outputs/full_results_plots")
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_results_plots(
        exploration_dir=args.exploration_dir,
        feature_dir=args.feature_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
