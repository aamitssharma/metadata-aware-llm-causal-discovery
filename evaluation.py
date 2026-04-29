import json
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

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
    edge_file: Path,
    no_edge_file: Path,
    model_name: str,
    gt_edges: set,
    alpha: float,
    beta: float,
    threshold: float,
) -> Dict[str, Any]:
    # Merge the two raw result files for one model into one per-edge ternary decision table.
    edge_results = _load_json(edge_file)
    no_edge_results = _load_json(no_edge_file)

    all_edge_keys = sorted(
        edge_key for edge_key in set(edge_results) | set(no_edge_results) if "->" in edge_key
    )
    per_edge_rows: List[Dict[str, Any]] = []

    for edge_key in all_edge_keys:
        source, target = edge_key.split("->", 1)
        edge_entry = edge_results.get(edge_key, {})
        no_edge_entry = no_edge_results.get(edge_key, {})

        p_edge = _statement_confidence(edge_entry)
        p_no_edge = _statement_confidence(no_edge_entry)
        edge_vote = _edge_label(edge_entry, threshold)
        no_edge_vote = _no_edge_label(no_edge_entry, threshold)
        final_prediction = _final_decision(edge_vote, no_edge_vote)
        ground_truth = "edge" if (source, target) in gt_edges else "no_edge"

        per_edge_rows.append(
            {
                "edge_key": edge_key,
                "source": source,
                "target": target,
                "ground_truth": ground_truth,
                "model_name": model_name,
                "p_edge": p_edge,
                "p_no_edge": p_no_edge,
                "edge_vote": edge_vote,
                "no_edge_vote": no_edge_vote,
                "final_prediction": final_prediction,
                "edge_notes": edge_entry.get("notes", []),
                "no_edge_notes": no_edge_entry.get("notes", []),
            }
        )

    adjusted = _adjusted_metrics(_adjusted_counts(per_edge_rows), alpha=alpha, beta=beta)
    return {
        "model_name": model_name,
        "edge_positive_class": "edge",
        "threshold": threshold,
        "hyperparameters": {
            "alpha": alpha,
            "beta": beta,
        },
        "adjusted_metrics": adjusted,
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
    alpha: float,
    beta: float,
    threshold: float,
) -> None:
    # One run folder may contain multiple models; evaluate each model pair separately.
    raw_config = _load_json(run_dir / "config.json")
    dataset_name = raw_config["dataset_name"]
    run_id = raw_config["run_id"]
    dataset_dir = raw_config["dataset_dir"]
    _, _, gt_edges = load_dataset(dataset_dir)

    raw_root = out_root / "RawLLMResults"
    relative_run_dir = run_dir.relative_to(raw_root)
    eval_dir = out_root / "EvaluatedResults" / relative_run_dir / f"thr_{threshold:g}__a_{alpha:g}__b_{beta:g}"
    eval_config = {
        "dataset_name": dataset_name,
        "dataset_dir": dataset_dir,
        "run_id": run_id,
        "raw_run_dir": str(run_dir),
        "prompt_family": raw_config.get("prompt_family", ""),
        "prompt_style": raw_config.get("prompt_style", ""),
        "variant_type": raw_config.get("variant_type", ""),
        "variant_name": raw_config.get("variant_name", ""),
        "metadata_file": raw_config.get("metadata_file", ""),
        "sampled_data_file": raw_config.get("sampled_data_file"),
        "threshold": threshold,
        "hyperparameters": {
            "alpha": alpha,
            "beta": beta,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(str(eval_dir / "config.json"), eval_config)

    slug_to_model = _model_lookup(raw_config.get("models", []))
    model_pairs = _pair_model_files(run_dir)

    for model_slug, pair in sorted(model_pairs.items()):
        edge_file = pair.get("edge")
        no_edge_file = pair.get("no_edge")
        if not edge_file or not no_edge_file:
            continue

        model_name = slug_to_model.get(model_slug, model_slug)
        evaluated = _evaluate_model(
            edge_file=edge_file,
            no_edge_file=no_edge_file,
            model_name=model_name,
            gt_edges=gt_edges,
            alpha=alpha,
            beta=beta,
            threshold=threshold,
        )
        evaluated.update(
            {
                "dataset_name": dataset_name,
                "run_id": run_id,
                "prompt_family": raw_config.get("prompt_family", ""),
                "prompt_style": raw_config.get("prompt_style", ""),
                "variant_type": raw_config.get("variant_type", ""),
                "variant_name": raw_config.get("variant_name", ""),
                "metadata_file": raw_config.get("metadata_file", ""),
                "sampled_data_file": raw_config.get("sampled_data_file"),
                "source_raw_files": {
                    "edge": str(edge_file),
                    "no_edge": str(no_edge_file),
                },
                "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        eval_path = eval_dir / f"ternaryEval__{model_slug}.json"
        save_json(str(eval_path), evaluated)
        edge_rows = []
        for row in evaluated["per_edge_results"]:
            edge_rows.append(
                {
                    "dataset_name": dataset_name,
                    "dataset_dir": dataset_dir,
                    "run_id": run_id,
                    "model_name": model_name,
                    "prompt_family": raw_config.get("prompt_family", ""),
                    "prompt_style": raw_config.get("prompt_style", ""),
                    "variant_type": raw_config.get("variant_type", ""),
                    "variant_name": raw_config.get("variant_name", ""),
                    "metadata_file": raw_config.get("metadata_file", ""),
                    "sampled_data_file": raw_config.get("sampled_data_file") or "",
                    "threshold": threshold,
                    **row,
                    "edge_notes": "|".join(str(note) for note in row.get("edge_notes", [])),
                    "no_edge_notes": "|".join(str(note) for note in row.get("no_edge_notes", [])),
                }
            )
        append_csv(edge_rows, str(eval_dir / "evaluated_edges.csv"))
        append_csv(edge_rows, str(out_root / "all_evaluated_edges.csv"))

        summary_row = _evaluation_summary_row(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            run_id=run_id,
            model_name=model_name,
            prompt_family=raw_config.get("prompt_family", ""),
            prompt_style=raw_config.get("prompt_style", ""),
            variant_type=raw_config.get("variant_type", ""),
            variant_name=raw_config.get("variant_name", ""),
            metadata_file=raw_config.get("metadata_file", ""),
            sampled_data_file=raw_config.get("sampled_data_file"),
            threshold=threshold,
            alpha=alpha,
            beta=beta,
            metrics=evaluated["adjusted_metrics"],
        )
        append_csv([summary_row], str(eval_dir / "evaluation_summary.csv"))
        append_csv([summary_row], str(out_root / "all_evaluation_summary.csv"))
        _upsert_manifest_entry(
            out_root / "all_evaluations.json",
            {
                "run_id": run_id,
                "dataset_name": dataset_name,
                "model_name": model_name,
                "prompt_family": raw_config.get("prompt_family", ""),
                "prompt_style": raw_config.get("prompt_style", ""),
                "variant_type": raw_config.get("variant_type", ""),
                "variant_name": raw_config.get("variant_name", ""),
                "threshold": threshold,
                "alpha": alpha,
                "beta": beta,
                "evaluation_timestamp": evaluated["evaluated_at"],
                "source_file": str(eval_path),
                "adjusted_metrics": evaluated["adjusted_metrics"],
            },
        )
        _print_metrics_summary(
            dataset_name=dataset_name,
            run_id=run_id,
            model_name=model_name,
            metrics=evaluated["adjusted_metrics"],
        )


def main() -> None:
    # evaluation.py is evaluation-only: it reads config.yaml and evaluates the listed run ids.
    config = load_project_config()
    experiment = config.get("experiment", {})
    evaluation = config.get("evaluation", {})

    out_root = Path(str(config.get("out_root", "outputs")))
    alpha_values = _normalize_float_list(evaluation.get("alpha", config.get("alpha")), 1.0)
    beta_values = _normalize_float_list(evaluation.get("beta", config.get("beta")), 1.0)
    threshold = float(evaluation.get("threshold", config.get("threshold", 0.7)))
    eval_run_ids = list(config.get("eval_run_ids", []))

    if not eval_run_ids:
        raise ValueError("config.yaml must define a non-empty 'eval_run_ids' list when evaluation is enabled.")

    raw_root = out_root / "RawLLMResults"

    _ = experiment  # reserved for future filtering by prompt family/style if needed
    for run_dir in _iter_run_dirs(raw_root, [str(run_id) for run_id in eval_run_ids]):
        for alpha, beta in product(alpha_values, beta_values):
            evaluate_run(
                run_dir=run_dir,
                out_root=out_root,
                alpha=alpha,
                beta=beta,
                threshold=threshold,
            )


if __name__ == "__main__":
    main()
