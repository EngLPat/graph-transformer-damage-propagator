#!/bin/bash
# Run the two single-hole constituents of extrap_twohole_diag
# (onehole_diag_a, onehole_diag_b). Same Abaqus invocation as run_extrap_cases.sh.
#
# The decks must already exist, e.g.
#   python -m src.preprocessing.generate_extrapolation_geometry --case onehole_diag_a
#   python -m src.preprocessing.generate_extrapolation_geometry --case onehole_diag_b

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CPUS="${CPUS:-24}"
SUB="$ROOT/Subroutine/usdfld3d.f90"
LOG_DIR="$ROOT/logs"
PROGRESS="$LOG_DIR/run_extrap_onehole_diag.log"
STUDIES=(extrap_onehole_diag_a extrap_onehole_diag_b)

mkdir -p "$LOG_DIR"
[ -f "$SUB" ] || { echo "ERROR: subroutine missing at $SUB"; exit 1; }

echo "================================================================" | tee -a "$PROGRESS"
echo "[run] diagonal single-hole cases at $(date)  cpus_per_job=$CPUS" | tee -a "$PROGRESS"
echo "================================================================" | tee -a "$PROGRESS"

run_one() {
    local study="$1"
    local job="plate_001"
    local geom="$ROOT/data/raw/$study/geometry/$job.inp"
    local sim_dir="$ROOT/data/raw/$study/simulations/$job"
    if [ ! -f "$geom" ]; then
        echo "[skip] $study: $geom missing" | tee -a "$PROGRESS"; return 1
    fi
    mkdir -p "$sim_dir"
    [ -e "$sim_dir/$job.inp" ] || ln -s "$geom" "$sim_dir/$job.inp"
    # Skip if a healthy ODB already exists
    if [ -f "$sim_dir/$job.odb" ] && [ "$(stat -c%s "$sim_dir/$job.odb")" -gt 1000000 ]; then
        echo "[skip] $study: already has .odb" | tee -a "$PROGRESS"; return 0
    fi
    cd "$sim_dir"
    echo "[$(date +%H:%M:%S)] BEGIN $study" | tee -a "$PROGRESS"
    abaqus job="$job" input="$job" user="$SUB" cpus=$CPUS mp_mode=mpi interactive \
        > "$job.stdout" 2> "$job.stderr"
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] OK    $study (rc=0)" | tee -a "$PROGRESS"
    else
        echo "[$(date +%H:%M:%S)] FAIL  $study (rc=$rc), see $sim_dir/$job.stderr" | tee -a "$PROGRESS"
    fi
    cd "$ROOT"
    return $rc
}

# The two diagonal constituents are cheap; run them together.
run_one "${STUDIES[0]}" & P1=$!
run_one "${STUDIES[1]}" & P2=$!
wait $P1 $P2

echo "[done] $(date)" | tee -a "$PROGRESS"
