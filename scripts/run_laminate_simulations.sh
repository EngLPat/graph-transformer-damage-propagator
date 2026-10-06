#!/bin/bash
# Parallel driver for the 400 simulations of the laminate study: runs MAXP
# Abaqus jobs at a time, each in its own data/raw/<study>/simulations/<plate>/
# directory (the layout build_cohesive_dataset.py expects).
#
# Tune MAXP and CPUS to the Abaqus licence and the machine. Default:
#   MAXP=2 x CPUS=24 = 48 cores, ~10 GB RAM.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STUDY="${STUDY:-laminate}"
N="${N:-400}"
CPUS="${CPUS:-24}"
MAXP="${MAXP:-2}"          # number of concurrent Abaqus jobs
SUB="$ROOT/Subroutine/usdfld3d.f90"

GEOM_DIR="$ROOT/data/raw/$STUDY/geometry"
SIM_ROOT="$ROOT/data/raw/$STUDY/simulations"
LOG_DIR="$ROOT/logs"
PROGRESS="$LOG_DIR/run_laminate_simulations.log"

mkdir -p "$LOG_DIR" "$SIM_ROOT"

if [ ! -f "$SUB" ]; then
    echo "ERROR: USDFLD subroutine not found at $SUB"; exit 1
fi
if [ ! -d "$GEOM_DIR" ] || [ ! -f "$GEOM_DIR/plate_001.inp" ]; then
    echo "ERROR: $GEOM_DIR missing or empty; generate the .inp files first."; exit 1
fi

echo "================================================================" | tee -a "$PROGRESS"
echo "[run] parallel Abaqus driver at $(date)"      | tee -a "$PROGRESS"
echo "      study=$STUDY  cpus_per_job=$CPUS  parallel=$MAXP" | tee -a "$PROGRESS"
echo "================================================================" | tee -a "$PROGRESS"

run_one() {
    local job="$1"
    local sim_dir="$SIM_ROOT/$job"
    mkdir -p "$sim_dir"
    # Link the .inp into the sim dir so abaqus finds it.
    if [ ! -e "$sim_dir/$job.inp" ]; then
        ln -s "$GEOM_DIR/$job.inp" "$sim_dir/$job.inp"
    fi
    cd "$sim_dir"
    echo "[$(date +%H:%M:%S)] BEGIN $job (pid $BASHPID)" | tee -a "$PROGRESS"
    abaqus job="$job" input="$job" user="$SUB" cpus=$CPUS mp_mode=mpi interactive \
        > "$job.stdout" 2> "$job.stderr"
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] OK    $job  (rc=0)" | tee -a "$PROGRESS"
    else
        echo "[$(date +%H:%M:%S)] FAIL  $job  (rc=$rc), see $sim_dir/$job.stderr" | tee -a "$PROGRESS"
    fi
}

running=0
for i in $(seq -f "%03g" 1 $N); do
    job="plate_$i"
    inp="$GEOM_DIR/$job.inp"
    if [ ! -f "$inp" ]; then
        echo "[skip] $job: $inp missing"      | tee -a "$PROGRESS"; continue
    fi
    # Skip if simulations/<plate>/<plate>.odb already exists at >1 MB
    odb="$SIM_ROOT/$job/$job.odb"
    if [ -f "$odb" ] && [ "$(stat -c%s "$odb")" -gt 1000000 ]; then
        echo "[skip] $job: already has .odb ($(stat -c%s "$odb") bytes)" | tee -a "$PROGRESS"
        continue
    fi

    run_one "$job" &
    running=$((running + 1))

    # Throttle: wait for ANY running job to finish before launching another
    if [ "$running" -ge "$MAXP" ]; then
        wait -n
        running=$((running - 1))
    fi
done

# Wait for the last batch.
wait
echo "================================================================" | tee -a "$PROGRESS"
echo "[done] $(date)" | tee -a "$PROGRESS"
