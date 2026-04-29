#!/usr/bin/env python3
"""
Generate synthetic data + graph JSON from a DISCRETE .bif file.

Features:
  - Read .bif via pgmpy (bnlearn-style BIF supported)
  - Validate CPTs sum to 1 per parent configuration
    * Optionally renormalize tiny rounding drift (e.g., 0.3333333*3 = 0.9999999)
  - Forward sampling to generate CSV data
  - Multiple seeds passed via CLI: --seeds 1 2 3 4 5
  - Outputs saved next to the .bif by default, or --outdir if provided

Outputs (default: same folder as .bif):
  - <stem>_graph.json
  - <stem>_seed<seed>.csv for each seed

Usage:
  python data/scripts/generate_data_graph_from_bif.py "path/to/net.bif" --n 20000 --seeds 1 2 3 4 5
  python data/scripts/generate_data_graph_from_bif.py "path/to/net.bif" --n 5000 --seeds 10 11 --outdir "some/output/folder"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling


def validate_cpds_sum_to_one(model, tol: float = 1e-9, renorm_tol: float = 1e-6) -> None:
    """
    Validate CPT columns sum to 1.0.

    Policy:
      - If |sum - 1| <= tol                 -> OK
      - If tol < |sum - 1| <= renorm_tol    -> Renormalize that column in-place (rounding drift)
      - If |sum - 1| > renorm_tol           -> FAIL (likely a real CPT issue)

    IMPORTANT: pgmpy stores CPDs as N-D tensors in cpd.values.
    cpd.get_values() returns a 2D view; if we renormalize, we must reshape back to original tensor shape.
    """
    errors: List[str] = []
    renormed: List[str] = []

    for cpd in model.get_cpds():
        # Work in 2D for "column sums": (child_states, parent_configs)
        values_2d = np.array(cpd.get_values(), dtype=float)
        col_sums = values_2d.sum(axis=0)

        # Negative probability check (hard fail)
        if np.any(values_2d < -tol):
            errors.append(f"{cpd.variable}: CPT contains negative probability values (min={values_2d.min()})")
            continue

        deltas = np.abs(col_sums - 1.0)
        bad_hard = np.where(deltas > renorm_tol)[0]
        bad_soft = np.where((deltas > tol) & (deltas <= renorm_tol))[0]

        # Renormalize "soft-bad" columns
        if bad_soft.size > 0:
            for j in bad_soft:
                s = col_sums[j]
                if s == 0.0:
                    errors.append(f"{cpd.variable}: column {j} sums to 0.0 (cannot renormalize)")
                else:
                    values_2d[:, j] = values_2d[:, j] / s

            # Write back preserving original N-D tensor shape
            orig_shape = np.array(cpd.values).shape
            try:
                cpd.values = values_2d.reshape(orig_shape)
            except Exception as e:
                errors.append(
                    f"{cpd.variable}: failed to reshape renormalized CPT back to {orig_shape}: {e}"
                )

            renormed.append(f"{cpd.variable}: renormalized {bad_soft.size}/{col_sums.size} columns")

        # Fail on "hard-bad" columns
        if bad_hard.size > 0:
            sample_bad = bad_hard[:10].tolist()
            errors.append(
                f"{cpd.variable}: {bad_hard.size}/{col_sums.size} CPT columns deviate > {renorm_tol} "
                f"(example cols: {sample_bad}, example sums: {col_sums[sample_bad]})"
            )

    if errors:
        msg = "CPT validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(msg)

    if renormed:
        print("⚠️  CPT renormalization applied (rounding drift):")
        for r in renormed:
            print("  - " + r)


def write_graph_json(out_path: Path, model) -> None:
    nodes = list(model.nodes())
    edges = [{"source": u, "target": v} for (u, v) in model.edges()]
    out_path.write_text(json.dumps({"nodes": nodes, "edges": edges}, indent=2), encoding="utf-8")


def forward_sample_df(model, n: int, seed: int) -> pd.DataFrame:
    """
    Forward sample from the BN.
    pgmpy signatures vary by version; we try common ones.

    NOTE: We DO NOT overwrite CPD shapes incorrectly (handled in validate()).
    """
    np.random.seed(seed)
    sampler = BayesianModelSampling(model)

    try:
        df = sampler.forward_sample(size=n, seed=seed, show_progress=False)
    except TypeError:
        try:
            df = sampler.forward_sample(size=n, seed=seed)
        except TypeError:
            df = sampler.forward_sample(size=n)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bif", help="Path to .bif file (quote it if it contains spaces/parentheses)")
    ap.add_argument("--n", type=int, default=20000, help="Rows per seed (default: 20000)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                    help="Seed list, e.g. --seeds 1 2 3 4 5")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="Tolerance for CPT sum check before renormalization (default: 1e-9)")
    ap.add_argument("--renorm_tol", type=float, default=1e-6,
                    help="If CPT column sums deviate by <= renorm_tol, auto-renormalize (default: 1e-6)")
    ap.add_argument("--outdir", type=str, default=None,
                    help="Output directory. If omitted, outputs go next to the .bif file.")
    args = ap.parse_args()

    bif_path = Path(args.bif).resolve()
    if not bif_path.exists():
        raise FileNotFoundError(f"Cannot find .bif file: {bif_path}")

    out_dir = Path(args.outdir).resolve() if args.outdir else bif_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = bif_path.stem

    # Read model
    reader = BIFReader(str(bif_path))
    model = reader.get_model()

    # Validate CPTs (and fix tiny rounding drift safely)
    validate_cpds_sum_to_one(model, tol=args.tol, renorm_tol=args.renorm_tol)

    # (Optional) Ensure model is still valid after any renorm
    model.check_model()

    # Write graph JSON
    graph_path = out_dir / f"{stem}_graph.json"
    write_graph_json(graph_path, model)
    print(f"✅ Wrote {graph_path}")

    # Stable column order
    try:
        col_order = list(model.topological_order())
    except Exception:
        col_order = list(model.nodes())

    # Generate datasets per seed
    for seed in args.seeds:
        df = forward_sample_df(model, n=args.n, seed=seed)

        # Reorder columns for consistency
        try:
            df = df[col_order]
        except Exception:
            pass

        csv_path = out_dir / f"{stem}_seed{seed}.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Wrote {csv_path}  (n={args.n}, seed={seed})")

    print("Done.")


if __name__ == "__main__":
    main()
