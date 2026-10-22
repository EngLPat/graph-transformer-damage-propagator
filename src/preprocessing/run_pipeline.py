"""Run the simulation stages of the preprocessing pipeline (Stages 1-2).

Usage::

    # Run both stages
    python src/preprocessing/run_pipeline.py

    # Run only the simulations for an existing study
    python src/preprocessing/run_pipeline.py --study <study> --stages 2

    # Override number of samples or seed
    python src/preprocessing/run_pipeline.py --n 50 --seed 123

    # Dry-run (Stage 2 only lists what it would do)
    python src/preprocessing/run_pipeline.py --dry-run

Stage overview
--------------
    Stage 1  generate_cohesive_laminate_uniform.py
                                    parametric .inp files + metadata
    Stage 2  run_simulations.py     run ABAQUS simulations

The training dataset is then built from the ODBs with
``python -m src.preprocessing.build_cohesive_dataset --study <study> ...``.

Both stages require ``abaqus`` on PATH: Stage 1 meshes with ``abaqus cae``
(noGUI), Stage 2 submits the jobs.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ALL_STAGES = [1, 2]


def _apply_study_paths(study_name: str):
    """Monkey-patch module-level path constants to add a study sub-directory.

    This keeps data from different case studies fully separated::

        data/raw/<study>/geometry/
        data/raw/<study>/simulations/
        data/processed/<study>/
    """
    import src.preprocessing.run_simulations as _rs
    _rs.GEOMETRY_DIR = PROJECT_ROOT / "data" / "raw" / study_name / "geometry"
    _rs.SIMULATIONS_DIR = PROJECT_ROOT / "data" / "raw" / study_name / "simulations"

    print(f"#  Study name    : {study_name}")
    print(f"#  Geometry      : data/raw/{study_name}/geometry/")
    print(f"#  Simulations   : data/raw/{study_name}/simulations/")
    print(f"#  Processed     : data/processed/{study_name}/")


def run_stage_1(n_samples, seed, layup=None, ply_thickness=None,
                study=None, elem_size=None, yt_range=None, **_):
    """Stage 1: Generate parametric geometry (.inp files).

    Uses ``generate_cohesive_laminate_uniform``: C3D8R plies + COH3D8
    interfaces, Poisson-DOF1 tie, uniform mesh (the recipe of every paper
    corpus).
    """
    import src.preprocessing.generate_cohesive_laminate_uniform as _gu
    # LAYUP_DEFAULT is the QI [45/-45/0/90] from the uniform generator
    if layup is not None:
        angles = [float(a) for a in layup.split(",")]
        t = ply_thickness if ply_thickness is not None else 0.125
        built_layup = [(a, round(t, 6)) for a in angles]
    else:
        built_layup = _gu.LAYUP_DEFAULT
    # Override the default element edge length if requested.  The
    # canonical recipe is 0.90 mm; pass --elem-size to vary.
    if elem_size is not None:
        _gu.ELEM_SIZE_DEFAULT = float(elem_size)
    # yt_range=None keeps the nominal Y_T in every deck
    _gu.generate_dataset_cohesive(
        n_samples=n_samples, seed=seed,
        layup=built_layup, study=study, material_model="usdfld",
        yt_range=yt_range,
    )


def run_stage_2(dry_run=False, parallel=4, user_subroutine=None, **_):
    """Stage 2: Run ABAQUS simulations.

    Pass ``user_subroutine`` (e.g. ``Subroutine/libstandardU.so``) for
    LaRC05/USDFLD plyres datasets; without it Abaqus aborts at
    increment 1 with "USER SUBROUTINE USDFLD IS MISSING". For built-in
    damage models (no subroutine) leave ``user_subroutine=None``.
    """
    from src.preprocessing.run_simulations import run_simulations
    run_simulations(parallel=parallel, dry_run=dry_run,
                    user_subroutine=user_subroutine)


STAGE_RUNNERS = {
    1: run_stage_1,
    2: run_stage_2,
}

STAGE_NAMES = {
    1: "Generate geometry",
    2: "Run simulations",
}


def main():
    parser = argparse.ArgumentParser(
        description="Run the simulation stages of the preprocessing pipeline (Stages 1-2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/preprocessing/run_pipeline.py                  # both stages\n"
            "  python src/preprocessing/run_pipeline.py --stages 2       # simulations only\n"
            "  python src/preprocessing/run_pipeline.py --n 100          # 100 samples\n"
            "  python src/preprocessing/run_pipeline.py --dry-run        # list-only mode\n"
        ),
    )
    parser.add_argument(
        "--stages", nargs="+", type=int, default=None,
        help="Which stages to run (default: both, 1 and 2).",
    )
    parser.add_argument(
        "--n", type=int, default=10,
        help="Number of geometry samples to generate (Stage 1, default: 10).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for geometry generation (Stage 1).",
    )
    parser.add_argument(
        "--parallel", type=int, default=4,
        help="Parallel ABAQUS jobs (Stage 2, default: 4).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Stage 2: list what would be processed without running.",
    )
    parser.add_argument(
        "--study", type=str, default=None,
        help="Study name: isolates all raw/processed data under "
             "sub-directories (e.g. --study rect_200).",
    )
    parser.add_argument(
        "--layup", type=str, default=None,
        help="Comma-separated fiber angles (Stage 1), e.g. '0,90,90,0'.",
    )
    parser.add_argument(
        "--ply-thickness", type=float, default=None,
        help="Per-ply thickness in mm (Stage 1). Default: 0.125.",
    )
    parser.add_argument(
        "--yt-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
        help="Stage 1. Vary the transverse tensile "
             "strength YT over [LO, HI] MPa, one stratified draw per sample, "
             "for the material-variability study. Omit it and every deck "
             "carries the nominal 60 MPa.",
    )
    parser.add_argument(
        "--user-subroutine", type=str, default=None,
        help="Stage 2: USDFLD user subroutine, a compiled library (.so, loaded "
             "through usub_lib_dir) or a source or object file (.f, .f90, .o, "
             "passed as Abaqus 'user=' option). REQUIRED for LaRC05 plyres "
             "datasets; without it sims abort with 'USER SUBROUTINE USDFLD "
             "IS MISSING'. Typical value: Subroutine/libstandardU.so.",
    )
    parser.add_argument(
        "--elem-size", type=float, default=None,
        help="Stage 1: in-plane element edge "
             "length in mm (default: 0.90, the canonical medium).",
    )
    args = parser.parse_args()

    stages = args.stages or ALL_STAGES

    # Validate stage numbers
    for s in stages:
        if s not in ALL_STAGES:
            print(f"ERROR: Unknown stage {s}. Valid stages: {ALL_STAGES}")
            sys.exit(1)

    print()
    print("#" * 70)
    print("#  PREPROCESSING PIPELINE")
    print("#" * 70)
    print(f"#  Stages to run : {stages}")
    print(f"#  Samples       : {args.n}")
    print(f"#  Seed          : {args.seed}")
    print(f"#  Dry run       : {args.dry_run}")
    if args.study:
        _apply_study_paths(args.study)
    print("#" * 70)
    print()

    timings = {}
    for s in sorted(stages):
        print(f"\n>>> STAGE {s}: {STAGE_NAMES[s].upper()}")
        print(f">>> {'=' * 60}\n")
        t0 = time.time()

        try:
            STAGE_RUNNERS[s](
                n_samples=args.n,
                seed=args.seed,
                parallel=args.parallel,
                dry_run=args.dry_run,
                layup=args.layup,
                ply_thickness=args.ply_thickness,
                yt_range=args.yt_range,
                study=args.study,
                elem_size=args.elem_size,
                user_subroutine=args.user_subroutine,
            )
        except SystemExit as e:
            if e.code != 0:
                print(f"\n!!! STAGE {s} exited with code {e.code}")
                sys.exit(e.code)
        except Exception as e:
            print(f"\n!!! STAGE {s} FAILED: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        elapsed = time.time() - t0
        timings[s] = elapsed
        print(f"\n>>> Stage {s} completed in {elapsed:.1f}s")

    # Summary
    print()
    print("#" * 70)
    print("#  PIPELINE SUMMARY")
    print("#" * 70)
    for s in sorted(stages):
        status = f"{timings[s]:.1f}s"
        print(f"#  Stage {s} ({STAGE_NAMES[s]:.<30s}) {status}")
    total = sum(timings.values())
    print(f"#  {'Total':.<38s} {total:.1f}s")
    print("#" * 70)


if __name__ == "__main__":
    main()
