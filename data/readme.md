# `data/` README

This folder contains:

1. **Discrete Bayesian Network datasets** (`.bif`) organized by network size, and
2. A **Python script** to generate **synthetic tabular datasets (CSV)** + a **graph JSON** from any discrete `.bif` file.

All Bayesian networks in this folder are sourced from **bnlearn** documentation: [https://www.bnlearn.com/documentation/](https://www.bnlearn.com/documentation/)

---

## Folder structure (high level)

```
data/
├─ Discrete Bayesian Networks/
│  ├─ Large Networks (50–100 nodes)/
│  │  ├─ hailfinder/
│  │
│  ├─ mediumNewteorks(20-50 nodes)/
│  │  ├─ alarm/
│  │  ├─ insurance
│  │
│  └─ SmallNetworks (<20 nodes)/
│     ├─ asia/
│     ├─ sach
│
├─ scripts/
│  └─ generate_data_graph_from_bif.py
│
└─ readme.md
```

> Note: Many paths contain spaces/parentheses, so **always quote the \*\***.bif\***\* path** in commands.

---

## What each dataset folder contains

Each dataset folder includes the original `.bif` file, and may also include generated artifacts after running the script.

**Common files inside a dataset folder:**

- `<name>.bif`
  The discrete Bayesian Network definition (variables + CPTs).
- `<name>_graph.json`
  Generated directed graph structure in `{ nodes, edges }` format.

- <name>\_categories.json
  Column schema + allowed categorical values for each variable (e.g., binary yes/no with canonical mapping yes→1, no→0) used to validate/normalize the generated CSV samples

- `<name>_seed<seed>.csv`
  Synthetic samples generated via forward sampling (one file per seed).
- `<name>_description.json` _(optional)_
  Human-readable metadata/notes about the dataset (if available).

Example:

```
<dataset>/
├─ <dataset>.bif
├─ asia_categories.json
├─ <dataset>_graph.json
├─ <dataset>_seed1.csv
├─ <dataset>_seed2.csv

├─ ...
└─ <dataset>_description.json
```

---

## Script: `scripts/generate_data_graph_from_bif.py`

### What it does

Given a discrete `.bif` Bayesian Network, the script:

- Parses the network using **pgmpy**
- Validates CPTs (**each conditional distribution sums to 1** per parent configuration)
- Performs **forward sampling** to generate synthetic data
- Writes a **graph JSON** (nodes + directed edges)
- Writes one **CSV per seed**

### Output location

By default (when `--outdir` is NOT provided), outputs are saved in the **same folder as the input \*\***.bif\*\*.

---

## Requirements

```bash
pip install pgmpy pandas numpy
```

Python 3 recommended.

---

## Usage

### Basic

```bash
python data/scripts/generate_data_graph_from_bif.py "path/to/net.bif" --n 20000 --seeds 1 2 3 4 5
```

### Example: Hailfinder

```bash
python data/scripts/generate_data_graph_from_bif.py \
  "data/Discrete Bayesian Networks/Large Networks (50–100 nodes)/hailfinder/hailfinder.bif" \
  --n 20000 \
  --seeds 1 2 3 4 5
```

### Example: Alarm

```bash
python data/scripts/generate_data_graph_from_bif.py \
  "data/Discrete Bayesian Networks/mediumNewteorks(20-50 nodes)/alarm/<YOUR_BIF_NAME>.bif" \
  --n 20000 \
  --seeds 1 2 3 4 5
```

### Optional: custom output folder

```bash
python data/scripts/generate_data_graph_from_bif.py \
  "path/to/net.bif" \
  --n 5000 \
  --seeds 10 11 \
  --outdir "some/output/folder"
```

---

## Notes / Tips

- If CPT validation fails, the script raises an error indicating which CPT columns don’t sum to 1.
- Multiple seeds are useful for reproducibility and for creating independent sampled datasets.
- Output file names always follow the input `.bif` stem automatically.
