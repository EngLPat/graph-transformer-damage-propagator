#!/bin/bash
# Run the three extrapolation simulations (no-hole / two-hole /
# stitched double plate) locally.  Same job layout as
# run_laminate_simulations.sh:
#   data/raw/<study>/simulations/plate_001/plate_001.odb
# 2 concurrent jobs x 24 CPUs (stitched runs last, alone, since it is 2x
# the size of the others).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CPUS="${CPUS:-24}"
SUB="$ROOT/Subroutine/libstandardU.so"
LOG_DIR="$ROOT/logs"
PROGRESS="$LOG_DIR/run_extrap_cases.log"
STUDIES=(extrap_nohole extrap_twohole extrap_stitched)

mkdir -p "$LOG_DIR"
[ -f "$SUB" ] || { echo "ERROR: subroutine missing at $SUB"; exit 1; }

echo "================================================================" | tee -a "$PROGRESS"
echo "[run] extrapolation cases at $(date)  cpus_per_job=$CPUS"          | tee -a "$PROGRESS"
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
    echo "usub_lib_dir='$(dirname "$SUB")'" > abaqus_v6.env   # Abaqus loads the compiled USDFLD from there
    echo "[$(date +%H:%M:%S)] BEGIN $study" | tee -a "$PROGRESS"
    abaqus job="$job" input="$job" cpus=$CPUS mp_mode=mpi interactive \
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

# small cases in parallel, big one after
run_one extrap_nohole &
P1=$!
run_one extrap_twohole &
P2=$!
wait $P1 $P2
run_one extrap_stitched

echo "[done] $(date)" | tee -a "$PROGRESS"
