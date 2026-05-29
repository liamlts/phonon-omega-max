"""CLI entry points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch  # noqa: F401  (must precede any libomp-touching dep on macOS)

from .data.load import load_phonons
from .eval.run import run_cgcnn_cv, run_gbdt_cv


def cmd_data(args) -> None:
    samples = load_phonons(cache_path=args.cache)
    print(f"loaded {len(samples)} samples → {args.cache}")


def cmd_train_gbdt(args) -> None:
    samples = load_phonons(cache_path=args.data_cache)
    summary = run_gbdt_cv(samples, cache_dir=args.cache_dir, metrics_dir=args.metrics_dir)
    (args.metrics_dir / "gbdt_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"GBDT mean MAE = {summary['mae_mean']:.1f} ± {summary['mae_std']:.1f} cm⁻¹")


def cmd_train_cgcnn(args) -> None:
    samples = load_phonons(cache_path=args.data_cache)
    summary = run_cgcnn_cv(
        samples,
        cache_dir=args.cache_dir,
        ckpt_dir=args.ckpt_dir,
        metrics_dir=args.metrics_dir,
        epochs=args.epochs,
    )
    (args.metrics_dir / "cgcnn_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"CGCNN mean MAE = {summary['mae_mean']:.1f} ± {summary['mae_std']:.1f} cm⁻¹")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m phonon_omegamax.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    dp = sub.add_parser("data")
    dp.add_argument("--cache", default="data/cache/phonons.parquet", type=Path)
    dp.set_defaults(func=cmd_data)

    gp = sub.add_parser("train-gbdt")
    gp.add_argument("--data-cache", default="data/cache/phonons.parquet", type=Path)
    gp.add_argument("--cache-dir", default="data/cache", type=Path)
    gp.add_argument("--metrics-dir", default="metrics", type=Path)
    gp.set_defaults(func=cmd_train_gbdt)

    cp = sub.add_parser("train-cgcnn")
    cp.add_argument("--data-cache", default="data/cache/phonons.parquet", type=Path)
    cp.add_argument("--cache-dir", default="data/cache", type=Path)
    cp.add_argument("--ckpt-dir", default="checkpoints", type=Path)
    cp.add_argument("--metrics-dir", default="metrics", type=Path)
    cp.add_argument("--epochs", default=200, type=int)
    cp.set_defaults(func=cmd_train_cgcnn)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
