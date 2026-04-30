import json
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from utils import append_csv, load_dataset, load_project_config, save_json


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _normalize_float_list(raw_value: Any, default: float) -> List[float]:
    if raw_value is None:
        return [float(default)]
    if isinstance(raw_value, list):
        values = [float(value) for value in raw_value]
    else:
        values = [float(raw_value)]
    cleaned = list(dict.fromkeys(values))
    if not cleaned:
        raise ValueError("Evaluation hyperparameter list cannot be empty.")
    return cleaned


def _float_slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "neg").replace(".", "p")


def _iter_run_dirs(raw_root: Path, run_ids: List[str]) -> List[Path]:
    # Inference now stores runs under a deeper experiment path, so discover config.json recursively.
    run_id_filter = set(run_ids)
    run_dirs: List[Path] = []
    if not raw_root.exists():
        return run_dirs

    for config_path in sorted(raw_root.rglob("config.json")):
        run_dir = config_path.parent
        try:
            raw_config = _load_json(config_path)
        except json.JSONDecodeError:
            continue

        if not raw_config.get("dataset_name") or not raw_config.get("run_id"):
            continue
        if run_id_filter and raw_config["run_id"] not in run_id_filter:
            continue
        run_dirs.append(run_dir)

    return run_dirs


def _statement_confidence(entry: Dict[str, Any]) -> float:
    # Keep the model's original confidence for the queried statement as-is.
    confidence = float(entry.get("confidence", 0.0))
    return max(0.0, min(1.0, confidence))


def _model_lookup(models: List[str]) -> Dict[str, str]:
    return {model.replace("/", "_"): model for model in models}


def _pair_model_files(run_dir: Path) -> Dict[str, Dict[str, Path]]:
    # Final ternary evaluation needs both prompt views for the same model.
    pairs: Dict[str, Dict[str, Path]] = {}
    for raw_file in sorted(run_dir.glob("*.json")):
        if raw_file.name == "config.json":
            continue
        if raw_file.name.startswith("edgeLLM__"):
            model_slug = raw_file.stem[len("edgeLLM__") :]
            pairs.setdefault(model_slug, {})["edge"] = raw_file
        elif raw_file.name.startswith("noEdgeLLM__"):
            model_slug = raw_file.stem[len("noEdgeLLM__") :]
            pairs.setdefault(model_slug, {})["no_edge"] = raw_file
    return pairs


def _edge_label(entry: Dict[str, Any], threshold: float) -> str:
    confidence = _statement_confidence(entry)
    return "edge" if confidence > threshold else "uncertain"


def _no_edge_label(entry: Dict[str, Any], threshold: float) -> str:
    confidence = _statement_confidence(entry)
    return "no_edge" if confidence > threshold else "uncertain"


def _final_decision(edge_label: str, no_edge_label: str) -> str:
    # Combine the edge-query and no-edge-query votes using the agreed 4-case table.
    decision_map = {
        ("edge", "uncertain"): "edge",
        ("edge", "no_edge"): "uncertain",
        ("uncertain", "uncertain"): "uncertain",
        ("uncertain", "no_edge"): "no_edge",
    }
    return decision_map[(edge_label, no_edge_label)]


def _adjusted_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    # UE = uncertain on a true edge, UN = uncertain on a true no-edge.
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "ue": 0, "un": 0}
    for row in rows:
        gt = row["ground_truth"]
        pred = row["final_prediction"]
        if gt == "edge" and pred == "edge":
            counts["tp"] += 1
        elif gt == "edge" and pred == "no_edge":
            counts["fn"] += 1
        elif gt == "edge" and pred == "uncertain":
            counts["ue"] += 1
        elif gt == "no_edge" and pred == "edge":
            counts["fp"] += 1
        elif gt == "no_edge" and pred == "no_edge":
            counts["tn"] += 1
        else:
            counts["un"] += 1
    return counts


def _adjusted_metrics(counts: Dict[str, int], alpha: float, beta: float) -> Dict[str, float]:
    # alpha and beta control how strongly uncertain predictions are penalized.
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    ue = counts["ue"]
    un = counts["un"]

    precision_adj = _safe_div(tp, tp + fp + alpha * un)
    recall_adj = _safe_div(tp, tp + fn + beta * ue)
    f1_adj = _safe_div(2 * precision_adj * recall_adj, precision_adj + recall_adj)
    accuracy_adj = _safe_div(tp + tn, tp + tn + fp + fn + alpha * un + beta * ue)

    return {
        **counts,
        "precision_adj": precision_adj,
        "recall_adj": recall_adj,
        "f1_adj": f1_adj,
        "accuracy_adj": accuracy_adj,
    }


def _adjusted_counts_from_frame(frame: pd.DataFrame) -> Dict[str, int]:
    if frame.empty:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "ue": 0, "un": 0}

    gt_edge = frame["ground_truth"].eq("edge")
    gt_no_edge = frame["ground_truth"].eq("no_edge")
    pred_edge = frame["final_prediction"].eq("edge")
    pred_no_edge = frame["final_prediction"].eq("no_edge")
    pred_uncertain = frame["final_prediction"].eq("uncertain")

    return {
        "tp": int((gt_edge & pred_edge).sum()),
        "fp": int((gt_no_edge & pred_edge).sum()),
        "fn": int((gt_edge & pred_no_edge).sum()),
        "tn": int((gt_no_edge & pred_no_edge).sum()),
        "ue": int((gt_edge & pred_uncertain).sum()),
        "un": int((gt_no_edge & pred_uncertain).sum()),
    }


def _print_metrics_summary(
    *,
    dataset_name: str,
    run_id: str,
    model_name: str,
    metrics: Dict[str, float],
) -> None:
    print(f"\nEvaluation complete: dataset={dataset_name} | run_id={run_id} | model={model_name}")
    print(f"  TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']} UE={metrics['ue']} UN={metrics['un']}")
    print(f"  Adj Accuracy : {metrics['accuracy_adj']:.4f}")
    print(f"  Adj Precision: {metrics['precision_adj']:.4f}")
    print(f"  Adj Recall   : {metrics['recall_adj']:.4f}")
    print(f"  Adj F1       : {metrics['f1_adj']:.4f}")


def _evaluation_summary_row(
    *,
    dataset_name: str,
    dataset_dir: str,
    run_id: str,
    model_name: str,
    temperature: float,
    prompt_family: str,
    prompt_style: str,
    variant_type: str,
    variant_name: str,
    metadata_file: str,
    sampled_data_file: str | None,
    threshold: float,
    alpha: float,
    beta: float,
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "dataset_dir": dataset_dir,
        "run_id": run_id,
        "model_name": model_name,
        "temperature": temperature,
        "prompt_family": prompt_family,
        "prompt_style": prompt_style,
        "variant_type": variant_type,
        "variant_name": variant_name,
        "metadata_file": metadata_file,
        "sampled_data_file": sampled_data_file or "",
        "threshold": threshold,
        "alpha": alpha,
        "beta": beta,
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
        "ue": metrics["ue"],
        "un": metrics["un"],
        "accuracy_adj": metrics["accuracy_adj"],
        "precision_adj": metrics["precision_adj"],
        "recall_adj": metrics["recall_adj"],
        "f1_adj": metrics["f1_adj"],
    }


def _evaluate_model(
    *,
    raw_frame: pd.DataFrame,
    model_name: str,
    gt_edges: set,
    threshold: float,
) -> Dict[str, Any]:
    # Merge edge/no-edge query rows into one per-edge ternary decision table.
    edge_frame = (
        raw_frame[raw_frame["query_mode"].eq("edge")]
        [["edge_key", "source", "target", "confidence", "notes"]]
        .rename(columns={"confidence": "p_edge", "notes": "edge_notes"})
    )
    no_edge_frame = (
        raw_frame[raw_frame["query_mode"].eq("no_edge")]
        [["edge_key", "source", "target", "confidence", "notes"]]
        .rename(columns={"confidence": "p_no_edge", "notes": "no_edge_notes"})
    )
    per_edge_frame = edge_frame.merge(
        no_edge_frame,
        on=["edge_key", "source", "target"],
        how="outer",
    )
    if not per_edge_frame.empty:
        per_edge_frame["p_edge"] = per_edge_frame["p_edge"].fillna(0.0).clip(0.0, 1.0)
        per_edge_frame["p_no_edge"] = per_edge_frame["p_no_edge"].fillna(0.0).clip(0.0, 1.0)
        per_edge_frame["edge_notes"] = per_edge_frame["edge_notes"].fillna("")
        per_edge_frame["no_edge_notes"] = per_edge_frame["no_edge_notes"].fillna("")
        per_edge_frame["model_name"] = model_name
        per_edge_frame["ground_truth"] = [
            "edge" if (source, target) in gt_edges else "no_edge"
            for source, target in zip(per_edge_frame["source"], per_edge_frame["target"])
        ]
        per_edge_frame["edge_vote"] = "uncertain"
        per_edge_frame.loc[per_edge_frame["p_edge"] > threshold, "edge_vote"] = "edge"
        per_edge_frame["no_edge_vote"] = "uncertain"
        per_edge_frame.loc[per_edge_frame["p_no_edge"] > threshold, "no_edge_vote"] = "no_edge"

        decision_map = {
            ("edge", "uncertain"): "edge",
            ("edge", "no_edge"): "uncertain",
            ("uncertain", "uncertain"): "uncertain",
            ("uncertain", "no_edge"): "no_edge",
        }
        per_edge_frame["final_prediction"] = [
            decision_map[(edge_vote, no_edge_vote)]
            for edge_vote, no_edge_vote in zip(per_edge_frame["edge_vote"], per_edge_frame["no_edge_vote"])
        ]
        per_edge_rows = per_edge_frame.to_dict(orient="records")
    else:
        per_edge_rows = []

    counts = _adjusted_counts_from_frame(per_edge_frame)
    return {
        "model_name": model_name,
        "edge_positive_class": "edge",
        "threshold": threshold,
        "counts": counts,
        "per_edge_results": per_edge_rows,
    }


def _upsert_manifest_entry(manifest_path: Path, new_entry: Dict[str, Any]) -> None:
    items: List[Dict[str, Any]] = []
    if manifest_path.exists():
        with manifest_path.open("r") as f:
            payload = json.load(f)
        items = payload.get("items", []) if isinstance(payload, dict) else []

    def _same_key(item: Dict[str, Any]) -> bool:
        return (
            item.get("run_id") == new_entry["run_id"]
            and item.get("dataset_name") == new_entry["dataset_name"]
            and item.get("model_name") == new_entry["model_name"]
            and item.get("temperature") == new_entry["temperature"]
            and item.get("prompt_family") == new_entry["prompt_family"]
            and item.get("prompt_style") == new_entry["prompt_style"]
            and item.get("variant_type") == new_entry["variant_type"]
            and item.get("variant_name") == new_entry["variant_name"]
            and item.get("alpha") == new_entry["alpha"]
            and item.get("beta") == new_entry["beta"]
            and item.get("threshold") == new_entry["threshold"]
        )

    items = [item for item in items if not _same_key(item)]
    items.append(new_entry)
    items.sort(
        key=lambda item: (
            item["dataset_name"],
            item["prompt_family"],
            item["prompt_style"],
            item["variant_type"],
            item["variant_name"],
            item["run_id"],
            item["model_name"],
            item.get("temperature", 0.0),
            item.get("threshold", 0.0),
            item.get("alpha", 0.0),
            item.get("beta", 0.0),
        )
    )
    save_json(
        str(manifest_path),
        {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "items": items,
        },
    )


def evaluate_run(
    *,
    run_dir: Path,
    out_root: Path,
    alpha_values: List[float],
    beta_values: List[float],
    threshold_values: List[float],
) -> None:
    # One run folder represents one model + prompt style and contains all temperatures/variants.
    raw_config = _load_json(run_dir / "config.json")
    dataset_name = raw_config["dataset_name"]
    run_id = raw_config["run_id"]
    dataset_dir = raw_config["dataset_dir"]
    _, _, gt_edges = load_dataset(dataset_dir)

    raw_root = out_root / raw_config.get("prompt_family", "")
    relative_run_dir = run_dir.relative_to(raw_root)
    eval_dir = out_root / "EvaluatedResults" / raw_config.get("prompt_family", "") / relative_run_dir
    eval_config = {
        "dataset_name": dataset_name,
        "dataset_dir": dataset_dir,
        "run_id": run_id,
        "raw_run_dir": str(run_dir),
        "temperatures": raw_config.get("temperatures", []),
        "prompt_family": raw_config.get("prompt_family", ""),
        "prompt_style": raw_config.get("prompt_style", ""),
        "variants": raw_config.get("variants", []),
        "thresholds": threshold_values,
        "hyperparameters": {
            "alpha": alpha_values,
            "beta": beta_values,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(str(eval_dir / "config.json"), eval_config)

    slug_to_model = _model_lookup(raw_config.get("models", []))
    model_slug = next(iter(slug_to_model), run_dir.parent.name)
    model_name = slug_to_model.get(model_slug, model_slug)
    raw_predictions_path = run_dir / "raw_predictions.csv"
    if not raw_predictions_path.exists():
        return

    raw_predictions = pd.read_csv(raw_predictions_path)
    grouping_columns = [
        "temperature",
        "variant_type",
        "variant_name",
        "metadata_file",
        "sampled_data_file",
    ]
    for group_values, raw_group in raw_predictions.groupby(grouping_columns, dropna=False, sort=True):
        temperature, variant_type, variant_name, metadata_file, sampled_data_file = group_values
        sampled_data_file = "" if pd.isna(sampled_data_file) else sampled_data_file
        slice_slug = f"temp_{_float_slug(temperature)}__{variant_type}__{variant_name}"
        for threshold in threshold_values:
            evaluated = _evaluate_model(
                raw_frame=raw_group,
                model_name=model_name,
                gt_edges=gt_edges,
                threshold=threshold,
            )
            evaluated_at = time.strftime("%Y-%m-%d %H:%M:%S")
            evaluated.update(
                {
                    "dataset_name": dataset_name,
                    "run_id": run_id,
                    "temperature": temperature,
                    "prompt_family": raw_config.get("prompt_family", ""),
                    "prompt_style": raw_config.get("prompt_style", ""),
                    "variant_type": variant_type,
                    "variant_name": variant_name,
                    "metadata_file": metadata_file,
                    "sampled_data_file": sampled_data_file,
                    "source_raw_files": {
                        "raw_predictions": str(raw_predictions_path),
                    },
                    "evaluated_at": evaluated_at,
                }
            )

            edge_rows = []
            for row in evaluated["per_edge_results"]:
                edge_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "dataset_dir": dataset_dir,
                        "run_id": run_id,
                        "model_name": model_name,
                        "temperature": temperature,
                        "prompt_family": raw_config.get("prompt_family", ""),
                        "prompt_style": raw_config.get("prompt_style", ""),
                        "variant_type": variant_type,
                        "variant_name": variant_name,
                        "metadata_file": metadata_file,
                        "sampled_data_file": sampled_data_file,
                        "threshold": threshold,
                        **row,
                        "edge_notes": str(row.get("edge_notes", "")),
                        "no_edge_notes": str(row.get("no_edge_notes", "")),
                    }
                )
            append_csv(edge_rows, str(eval_dir / f"evaluated_edges__{slice_slug}__thr_{threshold:g}.csv"))
            append_csv(edge_rows, str(out_root / "all_evaluated_edges.csv"))

            summary_rows = []
            metric_items = []
            for alpha, beta in product(alpha_values, beta_values):
                metrics = _adjusted_metrics(evaluated["counts"], alpha=alpha, beta=beta)
                summary_row = _evaluation_summary_row(
                    dataset_name=dataset_name,
                    dataset_dir=dataset_dir,
                    run_id=run_id,
                    model_name=model_name,
                    temperature=temperature,
                    prompt_family=raw_config.get("prompt_family", ""),
                    prompt_style=raw_config.get("prompt_style", ""),
                    variant_type=variant_type,
                    variant_name=variant_name,
                    metadata_file=metadata_file,
                    sampled_data_file=sampled_data_file,
                    threshold=threshold,
                    alpha=alpha,
                    beta=beta,
                    metrics=metrics,
                )
                summary_rows.append(summary_row)
                metric_items.append(
                    {
                        "threshold": threshold,
                        "alpha": alpha,
                        "beta": beta,
                        "adjusted_metrics": metrics,
                    }
                )

            evaluated["metric_grid"] = metric_items
            eval_path = eval_dir / f"ternaryEval__{slice_slug}__thr_{threshold:g}.json"
            save_json(str(eval_path), evaluated)
            append_csv(summary_rows, str(eval_dir / "evaluation_summary.csv"))
            append_csv(summary_rows, str(out_root / "all_evaluation_summary.csv"))

            for summary_row, metric_item in zip(summary_rows, metric_items):
                _upsert_manifest_entry(
                    out_root / "all_evaluations.json",
                    {
                        "run_id": run_id,
                        "dataset_name": dataset_name,
                        "model_name": model_name,
                        "temperature": temperature,
                        "prompt_family": raw_config.get("prompt_family", ""),
                        "prompt_style": raw_config.get("prompt_style", ""),
                        "variant_type": variant_type,
                        "variant_name": variant_name,
                        "threshold": summary_row["threshold"],
                        "alpha": summary_row["alpha"],
                        "beta": summary_row["beta"],
                        "evaluation_timestamp": evaluated_at,
                        "source_file": str(eval_path),
                        "adjusted_metrics": metric_item["adjusted_metrics"],
                    },
                )

            _print_metrics_summary(
                dataset_name=dataset_name,
                run_id=run_id,
                model_name=model_name,
                metrics=metric_items[0]["adjusted_metrics"],
            )


def main() -> None:
    # evaluation.py is evaluation-only: it reads config.yaml and evaluates the listed run ids.
    config = load_project_config()
    experiment = config.get("experiment", {})
    evaluation = config.get("evaluation", {})

    out_root = Path(str(config.get("out_root", "outputs")))
    alpha_values = _normalize_float_list(evaluation.get("alpha", config.get("alpha")), 1.0)
    beta_values = _normalize_float_list(evaluation.get("beta", config.get("beta")), 1.0)
    threshold_values = _normalize_float_list(evaluation.get("threshold", config.get("threshold")), 0.7)
    eval_run_ids = list(config.get("eval_run_ids", []))

    if not eval_run_ids:
        raise ValueError("config.yaml must define a non-empty 'eval_run_ids' list when evaluation is enabled.")

    raw_root = out_root / str(experiment.get("prompt_family", "metaData"))

    _ = experiment  # reserved for future filtering by prompt family/style if needed
    for run_dir in _iter_run_dirs(raw_root, [str(run_id) for run_id in eval_run_ids]):
        evaluate_run(
            run_dir=run_dir,
            out_root=out_root,
            alpha_values=alpha_values,
            beta_values=beta_values,
            threshold_values=threshold_values,
        )


if __name__ == "__main__":
    main()
