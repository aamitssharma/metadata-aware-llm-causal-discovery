import os
import time
import traceback
from typing import Any, Dict, List, Optional

from client import OpenRouterLLM
from prompt_loader import available_prompt_styles
from utils import (
    append_csv,
    load_csv_text,
    load_dataset,
    load_dotenv_file,
    load_metadata_schema,
    load_project_config,
    resolve_experiment_variants,
    save_json,
)


def _model_slug(model: str) -> str:
    return model.replace("/", "_")


def _mode_file_prefix(query_mode: str) -> str:
    return "edgeLLM" if query_mode == "edge" else "noEdgeLLM"


def _normalize_max_token_schedule(raw_value: Any) -> List[int]:
    if isinstance(raw_value, list):
        schedule = [int(value) for value in raw_value]
    else:
        schedule = [int(raw_value)]
    cleaned = sorted({value for value in schedule if value > 0})
    if not cleaned:
        raise ValueError("config.yaml must define at least one positive max_tokens value.")
    return cleaned


def _normalize_float_schedule(raw_value: Any, *, default: float, name: str) -> List[float]:
    if raw_value is None:
        values = [default]
    elif isinstance(raw_value, list):
        values = [float(value) for value in raw_value]
    else:
        values = [float(raw_value)]
    cleaned = list(dict.fromkeys(values))
    if not cleaned:
        raise ValueError(f"config.yaml must define at least one {name} value.")
    return cleaned


def _float_slug(value: float) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def _run_slice_slug(*, query_mode: str, temperature: float, variant_type: str, variant_name: str) -> str:
    return "__".join(
        [
            _mode_file_prefix(query_mode),
            f"temp_{_float_slug(temperature)}",
            variant_type,
            variant_name,
        ]
    )


def _normalize_prompt_styles(prompt_family: str, raw_value: Any) -> List[str]:
    available_styles = available_prompt_styles(prompt_family)
    if not available_styles:
        raise ValueError(f"No prompt styles found for prompt_family={prompt_family}")

    if raw_value is None:
        requested_styles = ["vanilla"]
    elif isinstance(raw_value, list):
        requested_styles = [str(value) for value in raw_value]
    else:
        requested_styles = [str(raw_value)]

    expanded: List[str] = []
    for style in requested_styles:
        style = style.strip()
        if style.lower() in {"all", "*"}:
            expanded.extend(available_styles)
        elif style:
            expanded.append(style)

    prompt_styles = list(dict.fromkeys(expanded))
    unsupported = [style for style in prompt_styles if style not in available_styles]
    if unsupported:
        raise ValueError(
            f"Unsupported prompt_style(s) for prompt_family={prompt_family}: {unsupported}. "
            f"Available: {available_styles}"
        )
    return prompt_styles


def _should_retry_with_more_tokens(error: Exception) -> bool:
    message = str(error).lower()
    retry_markers = [
        "length limit was reached",
        "eof while parsing",
        "finish_reason",
        "max tokens",
        "maximum context length",
        "too many tokens",
    ]
    return any(marker in message for marker in retry_markers)


def _save_model_outputs(
    *,
    run_dir: str,
    query_mode: str,
    temperature: float,
    variant_type: str,
    variant_name: str,
    raw_results: Dict[str, Dict[str, Any]],
) -> None:
    save_json(
        os.path.join(
            run_dir,
            "raw_json",
            f"{_run_slice_slug(query_mode=query_mode, temperature=temperature, variant_type=variant_type, variant_name=variant_name)}.json",
        ),
        raw_results,
    )


def _save_model_error(
    *,
    run_dir: str,
    out_root: str,
    dataset_name: str,
    dataset_dir: str,
    run_id: str,
    model: str,
    query_mode: str,
    prompt_family: str,
    prompt_style: str,
    variant_type: str,
    variant_name: str,
    metadata_file: str,
    sampled_data_file: Optional[str],
    temperature: float,
    max_tokens: int,
    error: Exception,
) -> None:
    error_payload = {
        "dataset_name": dataset_name,
        "dataset_dir": dataset_dir,
        "run_id": run_id,
        "model_name": model,
        "query_mode": query_mode,
        "prompt_family": prompt_family,
        "prompt_style": prompt_style,
        "variant_type": variant_type,
        "variant_name": variant_name,
        "metadata_file": metadata_file,
        "sampled_data_file": sampled_data_file or "",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(
        os.path.join(
            run_dir,
            "errors_json",
            f"{_run_slice_slug(query_mode=query_mode, temperature=temperature, variant_type=variant_type, variant_name=variant_name)}__error.json",
        ),
        error_payload,
    )
    append_csv([error_payload], os.path.join(run_dir, "errors.csv"))
    append_csv([error_payload], os.path.join(out_root, "all_errors.csv"))


def _raw_prediction_rows(
    *,
    raw_results: Dict[str, Dict[str, Any]],
    dataset_name: str,
    dataset_dir: str,
    run_id: str,
    model: str,
    query_mode: str,
    prompt_family: str,
    prompt_style: str,
    variant_type: str,
    variant_name: str,
    metadata_file: str,
    sampled_data_file: Optional[str],
    temperature: float,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for edge_key, entry in sorted(raw_results.items()):
        if "->" not in edge_key:
            continue
        source, target = edge_key.split("->", 1)
        rows.append(
            {
                "dataset_name": dataset_name,
                "dataset_dir": dataset_dir,
                "run_id": run_id,
                "model_name": model,
                "query_mode": query_mode,
                "prompt_family": prompt_family,
                "prompt_style": prompt_style,
                "variant_type": variant_type,
                "variant_name": variant_name,
                "metadata_file": metadata_file,
                "sampled_data_file": sampled_data_file or "",
                "temperature": temperature,
                "max_tokens": max_tokens,
                "edge_key": edge_key,
                "source": source,
                "target": target,
                "confidence": float(entry.get("confidence", 0.0)),
                "notes": "|".join(str(note) for note in entry.get("notes", [])),
            }
        )
    return rows


def _run_model_for_mode(
    *,
    llm: OpenRouterLLM,
    run_dir: str,
    model: str,
    schema: Dict[str, Any],
    query_mode: str,
    prompt_family: str,
    prompt_style: str,
    input_csv: str,
    dataset_name: str,
    dataset_dir: str,
    run_id: str,
    variant_type: str,
    variant_name: str,
    metadata_file: str,
    sampled_data_file: Optional[str],
    out_root: str,
    max_tokens: int,
    temperature: float,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Running model: {model} | mode: {query_mode} | max_tokens: {max_tokens}")
    print(f"{'=' * 60}")

    raw_results = llm.label_all_pairs(
        model=model,
        schema=schema,
        variables=list(schema.get("variables", {}).keys()),
        query_mode=query_mode,
        prompt_family=prompt_family,
        prompt_style=prompt_style,
        input_csv=input_csv,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    _save_model_outputs(
        run_dir=run_dir,
        query_mode=query_mode,
        temperature=temperature,
        variant_type=variant_type,
        variant_name=variant_name,
        raw_results=raw_results,
    )
    rows = _raw_prediction_rows(
        raw_results=raw_results,
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        run_id=run_id,
        model=model,
        query_mode=query_mode,
        prompt_family=prompt_family,
        prompt_style=prompt_style,
        variant_type=variant_type,
        variant_name=variant_name,
        metadata_file=metadata_file,
        sampled_data_file=sampled_data_file,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    append_csv(rows, os.path.join(run_dir, "raw_predictions.csv"))
    append_csv(rows, os.path.join(out_root, "all_raw_predictions.csv"))


def _build_run_dir(
    *,
    out_root: str,
    dataset_name: str,
    model: str,
    prompt_family: str,
    prompt_style: str,
    run_id: str,
) -> str:
    return os.path.join(
        out_root,
        prompt_family,
        run_id,
        dataset_name,
        _model_slug(model),
        prompt_style,
    )


def _save_run_config(
    *,
    run_dir: str,
    models: List[str],
    dataset_name: str,
    dataset_dir: str,
    out_root: str,
    run_id: str,
    max_tokens: List[int],
    temperatures: List[float],
    prompt_family: str,
    prompt_style: str,
    variants: List[Dict[str, Optional[str]]],
) -> None:
    save_json(
        os.path.join(run_dir, "config.json"),
        {
            "models": models,
            "dataset_name": dataset_name,
            "dataset_dir": dataset_dir,
            "out_root": out_root,
            "run_id": run_id,
            "max_tokens": max_tokens[0],
            "max_tokens_schedule": max_tokens,
            "temperatures": temperatures,
            "temperature": temperatures[0],
            "query_modes": ["edge", "no_edge"],
            "prompt_family": prompt_family,
            "prompt_style": prompt_style,
            "variants": variants,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def run_dataset(
    *,
    dataset_dir: str,
    models: List[str],
    out_root: str,
    max_tokens: List[int],
    temperatures: List[float],
    run_id: str,
    prompt_family: str,
    prompt_styles: List[str],
) -> None:
    dataset_name, _, gt_edges = load_dataset(dataset_dir)
    variants = resolve_experiment_variants(dataset_dir, prompt_family)
    llm = OpenRouterLLM()

    for prompt_style in prompt_styles:
        for model in models:
            run_dir = _build_run_dir(
                out_root=out_root,
                dataset_name=dataset_name,
                model=model,
                prompt_family=prompt_family,
                prompt_style=prompt_style,
                run_id=run_id,
            )

            _save_run_config(
                run_dir=run_dir,
                models=[model],
                dataset_name=dataset_name,
                dataset_dir=dataset_dir,
                out_root=out_root,
                run_id=run_id,
                max_tokens=max_tokens,
                temperatures=temperatures,
                prompt_family=prompt_family,
                prompt_style=prompt_style,
                variants=variants,
            )

            for variant in variants:
                schema = load_metadata_schema(variant["metadata_file"])
                input_csv = load_csv_text(variant["sampled_data_file"]) if variant["sampled_data_file"] else ""
                print(f"Loaded dataset: {dataset_name}")
                print(f"Prompt: {prompt_family}/{prompt_style}")
                print(f"Variant: {variant['variant_type']} | {variant['variant_name']}")
                print(f"Variables: {list(schema.get('variables', {}).keys())}")
                print(f"Ground truth edges: {len(gt_edges)}")

                for temperature in temperatures:
                    for query_mode in ["edge", "no_edge"]:
                        last_error: Optional[Exception] = None
                        for attempt_index, attempt_max_tokens in enumerate(max_tokens):
                            try:
                                if attempt_index > 0:
                                    print(
                                        f"Retrying {model} | temp: {temperature:g} | prompt: {prompt_style} | mode: {query_mode} with max_tokens={attempt_max_tokens}"
                                    )
                                _run_model_for_mode(
                                    llm=llm,
                                    run_dir=run_dir,
                                    model=model,
                                    schema=schema,
                                    query_mode=query_mode,
                                    prompt_family=prompt_family,
                                    prompt_style=prompt_style,
                                    input_csv=input_csv,
                                    dataset_name=dataset_name,
                                    dataset_dir=dataset_dir,
                                    run_id=run_id,
                                    variant_type=variant["variant_type"],
                                    variant_name=variant["variant_name"],
                                    metadata_file=variant["metadata_file"],
                                    sampled_data_file=variant["sampled_data_file"],
                                    out_root=out_root,
                                    max_tokens=attempt_max_tokens,
                                    temperature=temperature,
                                )
                                last_error = None
                                break
                            except Exception as exc:
                                last_error = exc
                                has_more_attempts = attempt_index < len(max_tokens) - 1
                                if has_more_attempts and _should_retry_with_more_tokens(exc):
                                    print(
                                        f"Retryable length/parsing failure for {model} | temp: {temperature:g} | prompt: {prompt_style} | mode: {query_mode} at max_tokens={attempt_max_tokens}"
                                    )
                                    print(f"Error: {exc}")
                                    continue
                                break

                        if last_error is not None:
                            print(f"Failed model: {model} | temp: {temperature:g} | prompt: {prompt_style} | mode: {query_mode}")
                            print(f"Error: {last_error}")
                            _save_model_error(
                                run_dir=run_dir,
                                out_root=out_root,
                                dataset_name=dataset_name,
                                dataset_dir=dataset_dir,
                                run_id=run_id,
                                model=model,
                                query_mode=query_mode,
                                prompt_family=prompt_family,
                                prompt_style=prompt_style,
                                variant_type=variant["variant_type"],
                                variant_name=variant["variant_name"],
                                metadata_file=variant["metadata_file"],
                                sampled_data_file=variant["sampled_data_file"],
                                temperature=temperature,
                                max_tokens=attempt_max_tokens,
                                error=last_error,
                            )
                            continue


def main() -> None:
    load_dotenv_file()
    config = load_project_config()
    experiment = config.get("experiment", {})

    models = list(config.get("models", []))
    dataset_dirs = list(config.get("datasets", config.get("dataset_dirs", [])))
    out_root = str(config.get("out_root", "outputs"))
    max_tokens = _normalize_max_token_schedule(config.get("max_tokens", [8000, 12000, 16000, 25000, 30000]))
    temperatures = _normalize_float_schedule(config.get("temperature"), default=0.0, name="temperature")
    run_tag = str(config.get("run_tag", "") or "")
    prompt_family = str(experiment.get("prompt_family", "metaData"))
    prompt_styles = _normalize_prompt_styles(prompt_family, experiment.get("prompt_style", "vanilla"))

    if not models:
        raise ValueError("config.yaml must define a non-empty 'models' list.")
    if not dataset_dirs:
        raise ValueError("config.yaml must define a non-empty 'datasets' list.")

    run_id = run_tag or f"run_{int(time.time())}"
    print(f"Run ID: {run_id}")
    print(f"Prompt styles: {', '.join(prompt_styles)}")
    print(f"Temperatures: {', '.join(f'{value:g}' for value in temperatures)}")

    for dataset_dir in dataset_dirs:
        try:
            run_dataset(
                dataset_dir=str(dataset_dir),
                models=models,
                out_root=out_root,
                max_tokens=max_tokens,
                temperatures=temperatures,
                run_id=run_id,
                prompt_family=prompt_family,
                prompt_styles=prompt_styles,
            )
        except Exception as exc:
            print(f"Dataset failed: {dataset_dir}")
            print(f"Error: {exc}")
            dataset_error = {
                "dataset_dir": str(dataset_dir),
                "run_id": run_id,
                "prompt_family": prompt_family,
                "prompt_styles": prompt_styles,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_json(
                os.path.join(out_root, "dataset_errors", f"{run_id}__{os.path.basename(str(dataset_dir))}.json"),
                dataset_error,
            )
            append_csv([dataset_error], os.path.join(out_root, "all_dataset_errors.csv"))
            continue

    print("\n" + "=" * 60)
    print("Run finished. Some items may have failed; check error logs if needed.")
    print(f"Run ID: {run_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
