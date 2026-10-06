#!/usr/bin/env python3
"""
Reconstruct a continuous train/val loss timeline across resumed sessions.

When a run's ``best_models/<model_name>.csv`` holds only part of its history
(for example after a resume with ``--resume-from``), the per-epoch losses of
the earlier sessions are still in the ``execution_log_train_*.txt`` files.

This tool reads all execution_log_train_*.txt under a result directory, parses
``Epoch N: Train=X Val=Y`` lines, stitches them by epoch number (later
sessions override earlier ones for overlapping epochs), and writes a single
loss plot with vertical dashed lines at each resume boundary.

It is read-only: existing ``loss_plot.png`` / ``loss_plot.pdf`` are not
overwritten. The new plot lands at ``training_results/loss_plot_resumed.png``
(plus a sibling .pdf).

Usage
-----
    python tools/plot_resume_history.py results/<run>

Flags:
    --out-name <stem>   override the output file stem (default: loss_plot_resumed)
    --no-pdf            skip the PDF sibling
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Matches "Epoch  37: Train=0.94733 Val=1.24716" with an optional " | rate=... | ..." tail.
EPOCH_RE = re.compile(
    r"^Epoch\s+(?P<epoch>\d+):\s*Train=(?P<train>[-+0-9.eE]+)\s+Val=(?P<val>[-+0-9.eE]+)"
)
RESUME_RE = re.compile(
    r"\[resume\][^\n]*start_epoch=(?P<start>\d+)"
)
# Matches "[resume] Will resume training from: results/<parent>/best_models/last_checkpoint.pt"
PARENT_RE = re.compile(
    r"\[resume\]\s+Will resume training from:\s+(?P<path>\S+)"
)


def parse_log(path: Path) -> Tuple[Dict[int, Tuple[float, float]], Optional[int], Optional[Path]]:
    """Return (epoch -> (train, val), resume_start, parent_dir).

    ``parent_dir`` is the directory of the checkpoint named in the
    ``[resume] Will resume training from:`` line, i.e. the grandparent of
    that ``<parent>/best_models/last_checkpoint.pt`` path. None if not
    a resumed session.
    """
    rows: Dict[int, Tuple[float, float]] = {}
    resume_start: Optional[int] = None
    parent_dir: Optional[Path] = None
    with path.open("r", errors="replace") as fh:
        for line in fh:
            m = EPOCH_RE.match(line)
            if m:
                rows[int(m.group("epoch"))] = (
                    float(m.group("train")),
                    float(m.group("val")),
                )
                continue
            r = RESUME_RE.search(line)
            if r is not None and resume_start is None:
                resume_start = int(r.group("start"))
            p = PARENT_RE.search(line)
            if p is not None and parent_dir is None:
                ckpt = Path(p.group("path"))
                # <parent>/best_models/<file>  →  <parent>
                if len(ckpt.parts) >= 2:
                    parent_dir = ckpt.parent.parent
    return rows, resume_start, parent_dir


def load_test_metrics(test_dir: Path) -> Optional[Dict[str, float]]:
    metrics_file = test_dir / "test_metrics.txt"
    if not metrics_file.exists():
        return None
    out: Dict[str, float] = {}
    fi_re = re.compile(r"^\s*FI integrated \(all\)\s+N=\s*\d+\s+R²=([-+0-9.eE]+)\s+RMSE=([-+0-9.eE]+)")
    with metrics_file.open("r") as fh:
        for line in fh:
            m = fi_re.match(line)
            if m:
                out["fi_r2"] = float(m.group(1))
                out["fi_rmse"] = float(m.group(2))
                break
    return out or None


def _collect_chain(result_dir: Path, seen: Optional[set] = None) -> List[Path]:
    """Walk back through resume parents, returning [oldest_ancestor, ..., result_dir].

    Detects parents by reading "[resume] Will resume training from: ..."
    lines in execution_log_train_*.txt. Guards against cycles via ``seen``.
    """
    if seen is None:
        seen = set()
    real = result_dir.resolve()
    if real in seen:
        return []
    seen.add(real)

    chain: List[Path] = [result_dir]
    logs = sorted(result_dir.glob("execution_log_train_*.txt"))
    parent: Optional[Path] = None
    for log_path in logs:
        _, _, p = parse_log(log_path)
        if p is not None:
            cand = p if p.is_absolute() else (result_dir.parent.parent / p)
            if not cand.exists():
                cand = (Path.cwd() / p) if not p.is_absolute() else p
            if cand.exists() and cand.is_dir():
                parent = cand
                break
    if parent is not None:
        chain = _collect_chain(parent, seen) + chain
    return chain


def stitch(result_dir: Path) -> Tuple[List[int], List[float], List[float], List[int]]:
    """Stitch losses across the whole resume chain, not just `result_dir`.

    Walks back through [resume] parent references in execution logs so a
    grandchild run's plot starts at epoch 0, not at its own first resumed
    epoch.
    """
    chain = _collect_chain(result_dir)
    logs: List[Path] = []
    for d in chain:
        logs.extend(sorted(d.glob("execution_log_train_*.txt")))
    if not logs:
        raise FileNotFoundError(f"No execution_log_train_*.txt under {result_dir} or its parents")

    merged: Dict[int, Tuple[float, float]] = {}
    resume_boundaries: List[int] = []

    for log_path in logs:
        rows, resume_start, _ = parse_log(log_path)
        if resume_start is not None:
            resume_boundaries.append(resume_start)
        # Later sessions override earlier ones for overlapping epochs.
        merged.update(rows)

    if not merged:
        raise ValueError(f"No 'Epoch N: Train=... Val=...' lines parsed under {result_dir}")

    epochs = sorted(merged.keys())
    train = [merged[e][0] for e in epochs]
    val = [merged[e][1] for e in epochs]
    return epochs, train, val, sorted(set(resume_boundaries))


def render(
    result_dir: Path,
    epochs: List[int],
    train: List[float],
    val: List[float],
    resume_boundaries: List[int],
    test_metrics: Optional[Dict[str, float]],
    out_name: str,
    write_pdf: bool,
) -> Path:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(epochs, train, color="tab:blue", linewidth=1.6, label="Train")
    ax.plot(epochs, val, color="tab:red", linewidth=1.6, label="Validation")

    # Mark best validation epoch.
    best_idx = int(np.argmin(val))
    ax.scatter(
        [epochs[best_idx]],
        [val[best_idx]],
        marker="*",
        s=180,
        color="orange",
        zorder=6,
        label=f"Best val: {val[best_idx]:.4f} @ ep {epochs[best_idx]}",
    )

    # Resume boundaries.
    for i, e in enumerate(resume_boundaries):
        ax.axvline(
            e,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label="Resume boundary" if i == 0 else None,
        )
        ax.text(
            e,
            ax.get_ylim()[1] if False else max(max(train), max(val)) * 0.98,
            f"resume @ {e}",
            rotation=90,
            verticalalignment="top",
            fontsize=8,
            color="gray",
        )

    # Test annotation.
    if test_metrics is not None:
        bits = []
        if "fi_r2" in test_metrics:
            bits.append(f"FI R²={test_metrics['fi_r2']:.4f}")
        if "fi_rmse" in test_metrics:
            bits.append(f"RMSE={test_metrics['fi_rmse']:.4f}")
        if bits:
            ax.text(
                0.99,
                0.02,
                "Test: " + "  ".join(bits),
                transform=ax.transAxes,
                fontsize=10,
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5"),
            )
    else:
        ax.text(
            0.99,
            0.02,
            "Test: missing (run --mode test to populate)",
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="bottom",
            color="firebrick",
            bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="firebrick"),
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{result_dir.name}: resumed-training loss timeline")
    ax.grid(True, linestyle="--", alpha=0.4)
    # Auto-log if losses span >2 orders of magnitude.
    span = max(max(train), max(val)) / max(min(train + val), 1e-12)
    if span > 100:
        ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    out_dir = result_dir / "training_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{out_name}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    if write_pdf:
        plt.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return png_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", type=Path, help="path/to/results/<run>")
    ap.add_argument("--out-name", default="loss_plot_resumed")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    result_dir: Path = args.result_dir.resolve()
    if not result_dir.is_dir():
        raise SystemExit(f"Not a directory: {result_dir}")

    epochs, train, val, resume_boundaries = stitch(result_dir)
    test_metrics = load_test_metrics(result_dir / "test_results")
    png_path = render(
        result_dir,
        epochs,
        train,
        val,
        resume_boundaries,
        test_metrics,
        args.out_name,
        write_pdf=not args.no_pdf,
    )
    print(f"  sessions stitched: {len(set([0] + resume_boundaries))}")
    print(f"  epochs covered:    {epochs[0]}..{epochs[-1]}  ({len(epochs)} unique)")
    print(f"  resume boundaries: {resume_boundaries if resume_boundaries else '(none)'}")
    if test_metrics is None:
        print("  test_metrics.txt:  MISSING (annotated on plot)")
    else:
        bits = ", ".join(f"{k}={v}" for k, v in test_metrics.items())
        print(f"  test metrics:      {bits}")
    print(f"  wrote: {png_path}")


if __name__ == "__main__":
    main()
