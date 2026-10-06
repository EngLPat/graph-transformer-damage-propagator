"""Run ABAQUS simulations in parallel for generated .inp geometries.

Stage 2 of the preprocessing pipeline.  Discovers all .inp files produced
by Stage 1, launches ABAQUS jobs, and organises the output .odb files into
per-simulation folders ready for ``build_cohesive_dataset.py``.

Each simulation is run as::

    abaqus job=<name> input=<inp> scratch=<tmp> interactive

Results layout
    data/raw/simulations/plate_001/plate_001.odb
    data/raw/simulations/plate_001/plate_001.inp   (copy)
    data/raw/simulations/plate_001/plate_001.dat
    ...

Usage
    python src/preprocessing/run_simulations.py               # default
    python src/preprocessing/run_simulations.py --parallel 8   # 8 workers
    python src/preprocessing/run_simulations.py --dry-run      # preview only
    python src/preprocessing/run_simulations.py --jobs plate_001 plate_003

Notes
    • Requires ABAQUS to be available on ``$PATH``.
    • On HPC, set ``--parallel`` to match the number of available cores.
    • Use ``--dry-run`` to verify file discovery without launching ABAQUS.
    • Failed simulations are logged but do not abort the remaining queue.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import argparse
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# PROJECT PATHS  (relative to repository root)
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GEOMETRY_DIR = PROJECT_ROOT / "data" / "raw" / "geometry"
SIMULATIONS_DIR = PROJECT_ROOT / "data" / "raw" / "simulations"
TEMP_DIR = PROJECT_ROOT / "temp" / "abaqus_scratch"

# ═══════════════════════════════════════════════════════════════════════════
# DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════
NUM_PARALLEL_JOBS = 4

# Extensions to keep from each simulation
KEEP_EXTENSIONS = [".odb", ".dat", ".msg", ".sta", ".log"]

# Extensions to clean up
TEMP_EXTENSIONS = [
    ".com", ".prt", ".sim", ".stt", ".res",
    ".pac", ".mdl", ".abq", ".lck", ".rec", ".023",
]


# ═══════════════════════════════════════════════════════════════════════════
# ABAQUS AVAILABILITY CHECK
# ═══════════════════════════════════════════════════════════════════════════

def check_abaqus_available():
    """Return True if ``abaqus`` is available on PATH."""
    try:
        result = subprocess.run(
            "abaqus information=system",
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE-JOB RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_single_simulation(inp_file, simulations_dir, temp_dir, project_root,
                          user_subroutine=None):
    """Run one ABAQUS simulation and organise outputs.

    Parameters
    ----------
    inp_file : Path
        Path to the .inp file (e.g. ``data/raw/geometry/plate_001.inp``).
    simulations_dir : Path
        Root output directory for simulation results.
    temp_dir : Path
        Root directory for ABAQUS scratch files.
    project_root : Path
        Repository root (used to catch stray ABAQUS files).
    user_subroutine : Path or None
        Optional path to a Fortran user subroutine (.f or .f90).

    Returns
    -------
    dict
        ``{"job_name": str, "success": bool, "odb_path": Path|None,
           "elapsed_s": float, "error": str|None}``
    """
    job_name = inp_file.stem
    result_folder = simulations_dir / job_name
    result_folder.mkdir(parents=True, exist_ok=True)

    job_temp_dir = temp_dir / job_name
    job_temp_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    error_msg = None

    try:
        print(f"  ▶ Starting: {job_name}", flush=True)
        user_clause = (f' user="{Path(user_subroutine).resolve()}"'
                       if user_subroutine else "")
        command = (
            f'abaqus job={job_name} input="{inp_file.resolve()}"'
            f'{user_clause}'
            f' scratch="{job_temp_dir.resolve()}" interactive'
        )
        process = subprocess.run(
            command,
            shell=True,
            cwd=str(job_temp_dir),
            capture_output=True,
            text=True,
        )

        if process.returncode != 0:
            error_msg = (
                f"ABAQUS exited with code {process.returncode}.  "
                f"stderr: {process.stderr[:500]}"
            )
            # Surface stderr/stdout immediately so the actual cause is visible
            # in the runner log. Truncated to 2 KB each to avoid flooding when
            # Abaqus dumps long verification banners.
            _stderr_tail = (process.stderr or "").strip()[-2000:]
            _stdout_tail = (process.stdout or "").strip()[-2000:]
            print(f"  ⚠ {job_name}: exit code {process.returncode}",
                  flush=True)
            if _stderr_tail:
                print(f"     stderr (last 2KB):\n{_stderr_tail}", flush=True)
            if _stdout_tail:
                print(f"     stdout (last 2KB):\n{_stdout_tail}", flush=True)

    except Exception as exc:
        error_msg = str(exc)
        print(f"  ✗ EXCEPTION: {job_name}: {error_msg}")

    # ── Organise outputs ──────────────────────────────────────────────
    # Look for result files in the scratch dir first, then project root
    odb_path = None
    for ext in KEEP_EXTENSIONS:
        for search_dir in [job_temp_dir, project_root]:
            src = search_dir / f"{job_name}{ext}"
            if src.exists():
                dst = result_folder / src.name
                shutil.move(str(src), str(dst))
                if ext == ".odb":
                    odb_path = dst
                break

    # Copy the .inp into the result folder for traceability (always overwrite)
    dst_inp = result_folder / inp_file.name
    if inp_file.exists():
        shutil.copy2(str(inp_file), str(dst_inp))

    # ── Cleanup ───────────────────────────────────────────────────────
    # Remove scratch directory
    try:
        if job_temp_dir.exists():
            shutil.rmtree(str(job_temp_dir))
    except Exception:
        pass

    # Remove stray files from project root
    for ext in TEMP_EXTENSIONS + KEEP_EXTENSIONS:
        stray = project_root / f"{job_name}{ext}"
        if stray.exists():
            try:
                stray.unlink()
            except Exception:
                pass

    elapsed = time.time() - t0

    # Success = ODB was produced, regardless of Abaqus exit code
    # (Abaqus often returns non-zero for warnings, .scn file issues, etc.)
    success = odb_path is not None

    status = "✓" if success else "✗"
    print(
        f"  {status} Finished: {job_name}  "
        f"({elapsed:.1f}s"
        f"{', ODB saved' if odb_path else ', NO ODB'}"
        f")",
        flush=True,
    )

    return {
        "job_name": job_name,
        "success": success,
        "odb_path": odb_path,
        "elapsed_s": elapsed,
        "error": error_msg,
    }


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL CLEANUP
# ═══════════════════════════════════════════════════════════════════════════

def cleanup_stray_files(directory):
    """Remove common ABAQUS temporary files from *directory*.

    Returns the number of files removed.
    """
    patterns = [
        "*.stt", "*.res", "*.pac", "*.mdl", "*.abq", "*.lck",
        "abaqus.rpy*", "*.rec", "*.023", "*.com", "*.prt",
    ]
    count = 0
    for pat in patterns:
        for f in Path(directory).glob(pat):
            try:
                f.unlink()
                count += 1
            except Exception:
                pass
    return count


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DRIVER
# ═══════════════════════════════════════════════════════════════════════════

def run_simulations(parallel=NUM_PARALLEL_JOBS, dry_run=False, jobs=None,
                    study=None, user_subroutine=None):
    """Run ABAQUS simulations for all (or selected) .inp files.

    Parameters
    ----------
    parallel : int
        Number of parallel workers.
    dry_run : bool
        If True, print what would be done without launching ABAQUS.
    jobs : list of str or None
        Optional list of job names to run (e.g. ``["plate_001", "plate_003"]``).
        If None, all .inp files in the geometry directory are processed.
    study : str or None
        Study subdirectory name (e.g. ``"coh_cp_0_90_s_test"``).
        If given, geometry is read from ``data/raw/geometry/<study>/``
        and results go to ``data/raw/simulations/<study>/``.
    user_subroutine : str or None
        Path to a Fortran user subroutine (.f or .f90) passed via ``user=``.

    Returns
    -------
    list of dict
        Per-job result dicts (see ``run_single_simulation``).
    """
    if study:
        # Match run_pipeline.py layout: data/raw/<study>/geometry/ and simulations/
        geometry_dir    = PROJECT_ROOT / "data" / "raw" / study / "geometry"
        simulations_dir = PROJECT_ROOT / "data" / "raw" / study / "simulations"
    else:
        geometry_dir    = GEOMETRY_DIR
        simulations_dir = SIMULATIONS_DIR

    print("=" * 70)
    print("STAGE 2: ABAQUS SIMULATION RUNNER")
    print("=" * 70)
    if study:
        print(f"  Study           : {study}")
    if user_subroutine:
        print(f"  User subroutine : {user_subroutine}")

    # ── Discover INP files ────────────────────────────────────────────
    if not geometry_dir.exists():
        print(f"✗ Geometry directory not found: {geometry_dir}")
        print("  Run Stage 1 (generate_cohesive_laminate_uniform.py) first.")
        sys.exit(1)

    all_inp_files = sorted(geometry_dir.glob("*.inp"))
    if not all_inp_files:
        print(f"✗ No .inp files found in {geometry_dir}")
        print("  Run Stage 1 (generate_cohesive_laminate_uniform.py) first.")
        sys.exit(1)

    # Filter to requested jobs if specified
    if jobs:
        job_set = set(jobs)
        inp_files = [f for f in all_inp_files if f.stem in job_set]
        not_found = job_set - {f.stem for f in inp_files}
        if not_found:
            print(f"  ⚠ Requested jobs not found: {sorted(not_found)}")
        if not inp_files:
            print("✗ No matching .inp files after filtering.")
            sys.exit(1)
    else:
        inp_files = all_inp_files

    print(f"  Geometry dir    : {geometry_dir}")
    print(f"  Output dir      : {simulations_dir}")
    print(f"  Scratch dir     : {TEMP_DIR}")
    print(f"  INP files found : {len(all_inp_files)}")
    print(f"  INP files to run: {len(inp_files)}")
    print(f"  Parallel workers: {parallel}")
    print(f"  Dry run         : {dry_run}")
    print()

    # ── List jobs ─────────────────────────────────────────────────────
    print("Jobs to run:")
    for i, f in enumerate(inp_files, 1):
        result_dir = simulations_dir / f.stem
        already_done = (result_dir / f"{f.stem}.odb").exists()
        status = " (ODB exists; will re-run)" if already_done else ""
        print(f"  {i:>3}. {f.stem}{status}")
    print()

    if dry_run:
        print("─" * 70)
        print("DRY RUN: no simulations launched.")
        print("─" * 70)
        print("\nOutput directories that would be created:")
        for f in inp_files:
            print(f"  {simulations_dir / f.stem}/")
        print("\nTo run for real, remove the --dry-run flag.")
        return []

    # ── Check ABAQUS ──────────────────────────────────────────────────
    print("Checking ABAQUS availability...")
    if not check_abaqus_available():
        print("✗ ABAQUS is not available on $PATH.")
        print("  Ensure ABAQUS is installed and 'abaqus' command is accessible.")
        sys.exit(1)
    print("  ✓ ABAQUS detected\n")

    # ── Prepare directories ───────────────────────────────────────────
    simulations_dir.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # ── Run simulations ───────────────────────────────────────────────
    print("─" * 70)
    print(f"Launching {len(inp_files)} simulations "
          f"({parallel} parallel workers)...")
    print("─" * 70)

    results = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        future_to_inp = {
            executor.submit(
                run_single_simulation,
                inp_file,
                simulations_dir,
                TEMP_DIR,
                PROJECT_ROOT,
                user_subroutine,
            ): inp_file
            for inp_file in inp_files
        }

        for future in as_completed(future_to_inp):
            inp_file = future_to_inp[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"  ✗ Unhandled exception for {inp_file.stem}: {exc}")
                results.append({
                    "job_name": inp_file.stem,
                    "success": False,
                    "odb_path": None,
                    "elapsed_s": 0,
                    "error": str(exc),
                })

    t_total = time.time() - t_start

    # ── Cleanup stray files ───────────────────────────────────────────
    cleaned = cleanup_stray_files(PROJECT_ROOT)
    if cleaned > 0:
        print(f"\n  Cleaned {cleaned} stray ABAQUS files from project root")

    # Remove empty scratch directory
    try:
        if TEMP_DIR.exists() and not any(TEMP_DIR.iterdir()):
            TEMP_DIR.rmdir()
            parent = TEMP_DIR.parent
            if parent.exists() and parent.name == "temp" and not any(parent.iterdir()):
                parent.rmdir()
    except Exception:
        pass

    # ── Summary ───────────────────────────────────────────────────────
    # Sort results by job name for readable output
    results.sort(key=lambda r: r["job_name"])
    succeeded = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print()
    print("=" * 70)
    print("STAGE 2 COMPLETE")
    print("=" * 70)
    print(f"  Total time      : {t_total:.1f}s")
    print(f"  Succeeded       : {len(succeeded)}/{len(results)}")
    print(f"  Failed          : {len(failed)}/{len(results)}")

    if succeeded:
        print("\n  Successful simulations:")
        for r in succeeded:
            print(f"    ✓ {r['job_name']}  ({r['elapsed_s']:.1f}s)")

    if failed:
        print("\n  Failed simulations:")
        for r in failed:
            err = r["error"] or "unknown error"
            print(f"    ✗ {r['job_name']}: {err[:80]}")

    print(f"\n  Results in: {simulations_dir}")
    print("=" * 70)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2: Run ABAQUS simulations for generated .inp files.",
    )
    parser.add_argument(
        "--parallel", type=int, default=NUM_PARALLEL_JOBS,
        help=f"Number of parallel ABAQUS workers (default: {NUM_PARALLEL_JOBS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be done without launching ABAQUS",
    )
    parser.add_argument(
        "--jobs", nargs="+", default=None,
        help="Run only specific jobs by name (e.g. plate_001 plate_003)",
    )
    parser.add_argument(
        "--study", type=str, default=None,
        help="Study subdirectory name (e.g. coh_cp_0_90_s_test). "
             "Reads from data/raw/geometry/<study>/, writes to data/raw/simulations/<study>/",
    )
    parser.add_argument(
        "--user-subroutine", type=str, default=None,
        help="Path to Fortran user subroutine (.f or .f90) passed via user= to ABAQUS",
    )
    args = parser.parse_args()

    run_simulations(
        parallel=args.parallel,
        dry_run=args.dry_run,
        jobs=args.jobs,
        study=args.study,
        user_subroutine=args.user_subroutine,
    )
