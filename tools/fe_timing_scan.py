"""Scan Abaqus .dat files for JOB TIME SUMMARY and report per-study FE-time stats.

For each study dir it reads every plate_*.dat, extracts TOTAL CPU TIME and
WALLCLOCK TIME (sec), and prints mean / median / std / min / max / n.

Run: [GTP_ROOT=<folder with data/>] python tools/fe_timing_scan.py
"""
from __future__ import annotations
import os
import re
from pathlib import Path
import statistics as st

# data/ is read under GTP_ROOT (default: this repository)
ROOT = Path(os.environ.get("GTP_ROOT", Path(__file__).resolve().parent.parent))

# (label, study dir under data/raw)
STUDIES = [
    ("single ply     ", "singleply"),
    ("laminate       ", "laminate"),
    ("laminate, Y_T  ", "laminate_yt_variability"),
]

CPU_RE = re.compile(r"TOTAL CPU TIME\s*\(SEC\)\s*=\s*([0-9.eE+]+)")
WALL_RE = re.compile(r"WALLCLOCK TIME\s*\(SEC\)\s*=\s*([0-9.eE+]+)")


def scan(study_dir: Path):
    cpu, wall = [], []
    for msg in study_dir.rglob("plate_*.dat"):
        try:
            txt = msg.read_text(errors="ignore")
        except Exception:
            continue
        mc = CPU_RE.findall(txt)
        mw = WALL_RE.findall(txt)
        if mc:
            cpu.append(float(mc[-1]))
        if mw:
            wall.append(float(mw[-1]))
    return cpu, wall


def summ(vals):
    if not vals:
        return "no data"
    return (f"n={len(vals):3d}  mean={st.mean(vals):9.1f}  median={st.median(vals):9.1f}  "
            f"std={(st.pstdev(vals) if len(vals)>1 else 0):8.1f}  "
            f"min={min(vals):8.1f}  max={max(vals):9.1f}")


def main():
    print("=" * 110)
    print("Abaqus FE time per study (seconds). CPU = total CPU across cores; WALL = wallclock.")
    print("=" * 110)
    for label, d in STUDIES:
        sd = ROOT / "data" / "raw" / d
        if not sd.exists():
            print(f"{label} [{d}]: DIR NOT FOUND")
            continue
        cpu, wall = scan(sd)
        print(f"\n{label} [{d}]")
        print(f"   CPU : {summ(cpu)}")
        print(f"   WALL: {summ(wall)}")
    print("\n" + "=" * 110)


if __name__ == "__main__":
    main()
