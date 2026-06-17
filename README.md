# LLM Causal Inference

Run causal-edge experiments across multiple datasets, models, prompt styles, and input conditions using OpenRouter.

The project now supports:
- `metaData` prompts
- `dataMetaData` prompts
- `vanilla`, `cot`, and `cot5Shot` prompt styles
- structured JSON outputs
- CSV files for later analysis

## Quick Start

Repo: `llmCausalInference`

From the `llmCausalInference` folder:

```bash
pip install -r requirements.txt
cp .env.example .env
```

Put your OpenRouter key into `.env`:

```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Then open [`config.yaml`](./config.yaml), choose:
- models
- datasets
- `prompt_family`
- `prompt_style`
- whether to run inference and/or evaluation

Run the program with:

```bash
bash run.sh
```

## How To Choose Experiments?

The main config lives in [`config.yaml`](./config.yaml).

Important fields:

```yaml
models:
  - openrouter/free

datasets:
  - data/DiscreteBayesianNetworks/SmallNetworks-20nodes/asia

experiment:
  prompt_family: metaData
  prompt_style: all
  run_inference: 1
  run_evaluation: 0

eval_run_ids:
  - run_1774191789

evaluation:
  threshold:
    - 0.65
    - 0.7
    - 0.75
  alpha: 1.0
  beta: 1.0
```

What these mean:
- `prompt_family: metaData`
  - use metadata-only prompts
  - runs clean metadata and noisy metadata variants
- `prompt_family: dataMetaData`
  - use prompts that need both metadata and sampled CSV data
  - runs clean metadata with clean and corrupt sampled-data variants
- `prompt_style`
  - `vanilla`, `cot`, `cot5Shot`, `all`, or a YAML list such as `[vanilla, cot]`
  - for `prompt_family: metaData`, `all` runs `vanilla`, `cot`, and `cot5Shot`
- `run_inference`
  - set to `1` to generate raw model outputs
- `run_evaluation`
  - set to `1` to evaluate existing raw runs listed in `eval_run_ids`
- `temperature`
  - accepts one value or a list of values
  - low values such as `0`, `0.25`, and `0.5` are best for repeatable causal judgments
- `evaluation.threshold`
  - accepts one value or a list of values
  - evaluation runs every threshold with every alpha/beta combination

Simple examples:

```yaml
# Metadata-only experiment
experiment:
  prompt_family: metaData
  prompt_style: all
  run_inference: 1
  run_evaluation: 0
```

```yaml
# Metadata + sampled-data experiment
experiment:
  prompt_family: dataMetaData
  prompt_style: cot
  run_inference: 1
  run_evaluation: 0
```

## Input Structure

For each dataset, the code expects the new folder layout like this:

```text
<dataset>/
  MetaData/
    L1_<dataset>.json
    L2_<dataset>.json
    L3_<dataset>.json
    L4_<dataset>.json
    L5_<dataset>_full_causal.json
  CITestsResults/
    <dataset>_filtered_seed1.csv
    ...
  <dataset>.bif
  <dataset>_description.json
  <dataset>_graph.json
  <dataset>_seed1.csv
  ...
```

Example for `asia`:
- metadata variants:
  - `MetaData/L1_asia.json`
  - `MetaData/L2_asia.json`
  - ...
- sampled data:
  - `asia_seed1.csv`
  - `asia_seed2.csv`
- CI test results:
  - `CITestsResults/asia_filtered_seed1.csv`

## Output Structure

Outputs are organized so you can easily tell:
- run id
- dataset
- model
- prompt family
- prompt style
- variant type
- exact variant name

Raw outputs:

```text
outputs/
  runs/
    <run_id>/
      <dataset_name>/
        <prompt_family>/
          <variant_name>/
            <prompt_style>/
              <model>/
                config.json
                raw_predictions.csv
                raw_json/
                  edgeLLM__temp_<temperature>.json
                  noEdgeLLM__temp_<temperature>.json
```

Evaluated outputs:

```text
outputs/
  evaluations/
    <run_id>/
      <dataset_name>/
        <prompt_family>/
          <variant_name>/
            <prompt_style>/
              <model>/
                config.json
                evaluated_edges__temp_<temperature>__thr_<threshold>.csv
                evaluation_summary.csv
                evaluated_json/
                  ternaryEval__temp_<temperature>__thr_<threshold>.json
```

Top-level aggregate files:
- [`outputs/all_raw_predictions.csv`](./outputs/all_raw_predictions.csv)
- [`outputs/all_evaluated_edges.csv`](./outputs/all_evaluated_edges.csv)
- [`outputs/all_evaluation_summary.csv`](./outputs/all_evaluation_summary.csv)
- [`outputs/all_evaluations.json`](./outputs/all_evaluations.json)

## JSON vs CSV

The project writes both:
- JSON for full detailed outputs
- CSV for easy analysis in pandas, spreadsheets, and plotting scripts

Use the CSV files when you want to compare:
- prompt styles
- models
- metadata variants
- sampled-data corruption levels
- datasets

## Typical Workflow

1. Edit [`config.yaml`](./config.yaml)
2. Set `experiment.run_inference: 1`
3. Run `bash run.sh`
4. Note the generated `run_id`
5. Put that `run_id` into `eval_run_ids`
6. Set `experiment.run_evaluation: 1`
7. Run `bash run.sh` again
8. Analyze the CSV outputs

## Notes

- `.env` is only for secrets like `OPENROUTER_API_KEY`
- `config.yaml` is the main experiment control file
- existing prompt files are used as-is; the pipeline now routes the right metadata/data inputs into them
