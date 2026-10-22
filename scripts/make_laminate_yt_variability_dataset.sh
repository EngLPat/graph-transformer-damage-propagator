#!/usr/bin/env bash
# =====================================================================
# Build the Y_T material-variability dataset.
#
# The 400-simulation eight-ply study regenerated with one difference: the
# transverse tensile strength YT is drawn per simulation over [51, 69] MPa
# (+-15% about the nominal 60 MPa). Seed and sample count match the baseline
# (seed 42, n 400) and the YT draw has its own generator, so plate_k is the
# baseline's plate_k with a different YT; this is checked after stage 1.
#
# Settings that reproduce the baseline decks:
#   --elem-size 0.90          the baseline mesh
#   --sym-bc --direct-bc      the baseline boundary conditions (not exposed by
#                             run_pipeline.py, so stage 1 calls the generator)
#   Subroutine/libstandardU.so  the compiled LaRC05 USDFLD subroutine (*Depvar 10)
#
# Resources: about 17,000 s of CPU per simulation on 8 threads (~37 h for 400
# jobs at 6 in parallel) and about 105 GB of disk.
#
# Usage:
#   bash scripts/make_laminate_yt_variability_dataset.sh 2>&1 | tee logs/yt_dataset_$(date +%Y%m%d_%H%M).log
# =====================================================================
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BASELINE="laminate"
STUDY="laminate_yt_variability"
N_SIMS=400
SEED=42                 # as the baseline, so the geometries are paired
PARALLEL_JOBS=6         # 6 jobs x 8 threads
ELEM_SIZE=0.90          # the baseline mesh
LAYUP="45,-45,0,90"
PLY_THICKNESS=0.375
YT_LO=51                # nominal 60 MPa, -15%
YT_HI=69                # nominal 60 MPa, +15%
USDFLD="Subroutine/libstandardU.so"

cd "$PROJECT_DIR"
for CONDA_SH in ~/miniconda3/etc/profile.d/conda.sh ~/anaconda3/etc/profile.d/conda.sh; do
    [ -f "$CONDA_SH" ] && { source "$CONDA_SH"; break; }
done
conda activate gtp
mkdir -p logs

FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
echo "free disk: ${FREE_GB} GB"
if [ "$FREE_GB" -lt 120 ]; then
    echo "ABORT: under 120 GB free and this study needs about 105 GB."
    echo "       Free some disk space first."
    exit 1
fi

# ---------------------------------------------------------- stage 1 --------
echo "=== STAGE 1: decks, $N_SIMS samples, YT in [$YT_LO, $YT_HI] MPa  $(date -Is)"
python -m src.preprocessing.generate_cohesive_laminate_uniform \
    --study "$STUDY" \
    --n "$N_SIMS" \
    --seed "$SEED" \
    --layup "$LAYUP" \
    --ply-thickness "$PLY_THICKNESS" \
    --elem-size "$ELEM_SIZE" \
    --sym-bc --direct-bc \
    --yt-range "$YT_LO" "$YT_HI"

# ---------------------------------------------------------- checks ---------
# Pairing checks against the baseline (mesh counts, geometry parameters,
# boundary conditions, YT band); a failure stops the run before any Abaqus job.
echo "=== pairing and correctness checks"
python - "$BASELINE" "$STUDY" "$YT_LO" "$YT_HI" <<'PY'
import sys, re, torch, numpy as np
base, study, lo, hi = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
ok = True

def mesh_counts(path):
    n = e = 0; mode = None
    for line in open(path, errors="ignore"):
        s = line.strip().lower()
        if s.startswith("*node"):    mode = "n"; continue
        if s.startswith("*element"): mode = "e"; continue
        if s.startswith("*"):        mode = None; continue
        if not line.strip():         continue
        if mode == "n": n += 1
        elif mode == "e": e += 1
    return n, e

# 1. mesh identical to the baseline for the first, middle and last plate
for k in ("001", "200", "400"):
    ca = mesh_counts(f"data/raw/{base}/geometry/plate_{k}.inp")
    cb = mesh_counts(f"data/raw/{study}/geometry/plate_{k}.inp")
    good = ca == cb
    ok &= good
    print(f"  plate_{k} mesh  baseline {ca}  new {cb}   {'OK' if good else 'MISMATCH'}")

# 2. geometry parameters identical to the baseline, all 400 of them
pb = torch.load(f"data/processed/{base}/plate_geometry_data.pt",
                weights_only=False)["params"].numpy()
pn = torch.load(f"data/processed/{study}/plate_geometry_data.pt",
                weights_only=False)["params"].numpy()
d = float(np.abs(pb - pn).max())
print(f"  geometry parameters, max abs difference over 400 x 6: {d:.3e}   "
      f"{'OK' if d == 0 else 'NOT PAIRED'}")
ok &= (d == 0)

# 3. the deck really is direct-BC and sym-BC, as the baseline is
txt1 = open(f"data/raw/{study}/geometry/plate_001.inp", errors="ignore").read()
bc_ok = ("Set-RightDirect" in txt1) and ("TopFaceZ" in txt1) and ("\n*Coupling" not in txt1)
print(f"  boundary conditions: direct-BC and sym-BC present, no coupling: {bc_ok}")
ok &= bc_ok

# 4. YT varies, stays inside the band, and every sample is distinct
yts = []
for k in range(1, 401):
    t = open(f"data/raw/{study}/geometry/plate_{k:03d}.inp", errors="ignore").read()
    m = re.search(r"\*Property Table, type=LARC05PROPERTIES\s*\n([^\n]*)", t)
    yts.append(float(m.group(1).split(",")[6]))
yts = np.array(yts)
inside = bool(((yts >= lo - 1e-6) & (yts <= hi + 1e-6)).all())
print(f"  YT: n={len(yts)} unique={len(np.unique(yts))} min={yts.min():.2f} "
      f"max={yts.max():.2f} mean={yts.mean():.2f}  inside band: {inside}")
ok &= inside and len(np.unique(yts)) == 400

print("  ALL CHECKS PASSED" if ok else "  CHECKS FAILED")
sys.exit(0 if ok else 1)
PY

# ---------------------------------------------------------- stage 2 --------
echo "=== STAGE 2: Abaqus, $N_SIMS jobs, $PARALLEL_JOBS parallel  $(date -Is)"
# The LaRC05 USDFLD is mandatory. Without it Abaqus aborts at increment 1 and
# every .odb holds only the t=0 frame.
python src/preprocessing/run_pipeline.py \
    --study "$STUDY" \
    --stages 2 \
    --parallel "$PARALLEL_JOBS" \
    --user-subroutine "$USDFLD"

# ---------------------------------------------------------- stage 3 --------
echo "=== STAGE 3: dataset build  $(date -Is)"
# --include-damage-feature adds the damage state as an input and
# --include-yt-feature gives every node its sample's YT (16 node features);
# --resample-dt 0.01 gives the uniform time step.
python -m src.preprocessing.build_cohesive_dataset \
    --study "$STUDY" \
    --include-damage-feature \
    --include-yt-feature \
    --out "${STUDY}_dmgstate_ytfeat_dataset.pt" \
    --resample-dt 0.01

echo "=== DONE  $(date -Is)"
ls -la "data/processed/$STUDY/"
echo
echo "Training dataset:"
echo "  data/processed/$STUDY/${STUDY}_dmgstate_ytfeat_dataset_uniform_dt0.01.pt"
