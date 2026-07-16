"""Create human-readable context reports for trained fraud models."""

from __future__ import annotations

import argparse
import io
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

import fraud_model_training


def run_model_context(
    model_dir: str | Path = "outputs/full_model_training",
    output_dir: str | Path = "outputs/full_model_context",
) -> dict[str, Any]:
    model_path = Path(model_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    metrics = read_csv(model_path / "model_comparison_metrics.csv")
    confusion = read_csv(model_path / "confusion_matrices.csv")
    predictions = read_csv(model_path / "validation_predictions.csv")
    tree_rules = read_csv(model_path / "decision_tree_rules.csv")
    run_summary = read_json(model_path / "run_summary.json")

    training_summary = summarize_training(metrics, confusion, predictions, run_summary)
    training_summary.to_csv(output / "training_performance_context.csv", index=False)

    tree_impacts = decision_tree_decision_impacts(model_path / "decision_tree_model.pkl")
    tree_impacts.to_csv(output / "decision_tree_decision_impacts.csv", index=False)

    model_report = context_markdown(
        metrics=metrics,
        confusion=confusion,
        predictions=predictions,
        tree_rules=tree_rules,
        tree_impacts=tree_impacts,
        run_summary=run_summary,
    )
    (output / "model_context_report.md").write_text(model_report)

    summary = {
        "model_dir": str(model_dir),
        "outputs": [
            "training_performance_context.csv",
            "decision_tree_decision_impacts.csv",
            "model_context_report.md",
        ],
    }
    (output / "model_context_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def summarize_training(
    metrics: pd.DataFrame,
    confusion: pd.DataFrame,
    predictions: pd.DataFrame,
    run_summary: dict[str, Any],
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=["item", "value", "context"])
    rows = []
    for metric in ["roc_auc", "pr_auc", "precision", "recall", "f1"]:
        if metric in metrics:
            best = metrics.sort_values(metric, ascending=False).iloc[0]
            rows.append(
                {
                    "item": f"best_{metric}",
                    "value": best.get(metric),
                    "context": f"{best.get('model')} had the highest {metric}.",
                }
            )
    if not predictions.empty and "actual" in predictions:
        positives = int((predictions["actual"] == 1).sum())
        rows.append(
            {
                "item": "validation_positive_rate",
                "value": positives / len(predictions) if len(predictions) else 0.0,
                "context": f"Validation rows contained {positives} fraud positives out of {len(predictions)} rows.",
            }
        )
    for key in ["rows", "train_rows", "validation_rows", "features_used"]:
        if key in run_summary:
            rows.append(
                {
                    "item": key,
                    "value": run_summary[key],
                    "context": f"Run summary reported {key}.",
                }
            )
    if not confusion.empty:
        for _, row in confusion.iterrows():
            tp = int(row.get("tp", 0))
            fp = int(row.get("fp", 0))
            fn = int(row.get("fn", 0))
            tn = int(row.get("tn", 0))
            rows.append(
                {
                    "item": f"{row.get('model')}_confusion_summary",
                    "value": tp + tn,
                    "context": (
                        f"{row.get('model')} produced TP={tp}, FP={fp}, "
                        f"FN={fn}, TN={tn} at the 0.5 threshold."
                    ),
                }
            )
    return pd.DataFrame(rows)


def decision_tree_decision_impacts(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "path_before_decision",
                "decision",
                "parent_prediction",
                "child_prediction",
                "prediction_delta",
                "child_samples",
                "child_fraud_rate",
            ]
        )
    model = load_pickle_with_main_mapping(path)
    tree = getattr(model, "tree_", None)
    if not tree:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    collect_decision_impacts(tree, [], rows)
    return pd.DataFrame(rows).sort_values("prediction_delta", key=lambda s: s.abs(), ascending=False)


def collect_decision_impacts(
    node: dict[str, Any],
    path: list[str],
    rows: list[dict[str, Any]],
) -> None:
    if node.get("is_leaf", True):
        return
    parent_prediction = float(node.get("prediction", 0.0))
    split = node["split"]
    branches = [
        (fraud_model_training.format_tree_condition(split, positive=True), node["left"]),
        (fraud_model_training.format_tree_condition(split, positive=False), node["right"]),
    ]
    for condition, child in branches:
        child_prediction = float(child.get("prediction", 0.0))
        rows.append(
            {
                "path_before_decision": " AND ".join(path) if path else "root",
                "decision": condition,
                "parent_prediction": parent_prediction,
                "child_prediction": child_prediction,
                "prediction_delta": child_prediction - parent_prediction,
                "child_samples": int(child.get("samples", 0)),
                "child_fraud_rate": float(child.get("fraud_rate", 0.0)),
            }
        )
        collect_decision_impacts(child, path + [condition], rows)


def context_markdown(
    metrics: pd.DataFrame,
    confusion: pd.DataFrame,
    predictions: pd.DataFrame,
    tree_rules: pd.DataFrame,
    tree_impacts: pd.DataFrame,
    run_summary: dict[str, Any],
) -> str:
    lines = ["# Fraud Model Context Report", ""]
    lines.append("## Run Summary")
    lines.append("")
    if run_summary:
        for key, value in run_summary.items():
            if key != "outputs":
                lines.append(f"- **{key}**: {value}")
    else:
        lines.append("_No run summary was found._")
    lines.append("")

    lines.append("## Model Performance")
    lines.append("")
    lines.append(markdown_table(metrics))
    lines.append("")

    lines.append("## Confusion Matrix Context")
    lines.append("")
    if confusion.empty:
        lines.append("_No confusion matrix output was found._")
    else:
        for _, row in confusion.iterrows():
            lines.append(
                f"- **{row.get('model')}**: TP={int(row.get('tp', 0))}, "
                f"FP={int(row.get('fp', 0))}, FN={int(row.get('fn', 0))}, "
                f"TN={int(row.get('tn', 0))} at threshold 0.5."
            )
    lines.append("")

    lines.append("## Validation Prediction Context")
    lines.append("")
    if predictions.empty or "actual" not in predictions:
        lines.append("_No validation predictions were found._")
    else:
        positives = int((predictions["actual"] == 1).sum())
        lines.append(f"- Validation rows: {len(predictions)}")
        lines.append(f"- Fraud positives in validation rows: {positives}")
        lines.append(f"- Fraud-positive rate: {positives / len(predictions):.6f}")
    lines.append("")

    lines.append("## Decision Tree Leaf Context")
    lines.append("")
    lines.append(markdown_table(tree_rules.head(20)))
    lines.append("")

    lines.append("## Decision Tree Decision Impacts")
    lines.append("")
    lines.append(
        "Each row shows how one branch changes the predicted fraud probability "
        "relative to the parent node."
    )
    lines.append("")
    lines.append(markdown_table(tree_impacts.head(30)))
    lines.append("")
    return "\n".join(lines)


class MainMappingUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and hasattr(fraud_model_training, name):
            return getattr(fraud_model_training, name)
        return super().find_class(module, name)


def load_pickle_with_main_mapping(path: Path) -> Any:
    data = path.read_bytes()
    return MainMappingUnpickler(io.BytesIO(data)).load()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


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
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create model context reports.")
    parser.add_argument("--model-dir", default="outputs/full_model_training")
    parser.add_argument("--output-dir", default="outputs/full_model_context")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_model_context(model_dir=args.model_dir, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
