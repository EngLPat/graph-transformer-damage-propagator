#!/bin/bash
# Single-ply dataset: 200 simulations of a 90-degree ply (0.125 mm) with a
# uniform 1.5 mm mesh and a kinematic coupling on the loaded edge (the
# generator's default boundary conditions). Generates the decks, checks them
# (tools/check_singleply_decks.py), smoke-runs plate_001, runs the remaining
# simulations, and builds the dataset with the damage feature at dt = 0.01.
# Usage, in the activated environment: ./scripts/make_singleply_dataset.sh
set -eo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
STUDY=singleply
PY=${PYTHON:-python}
SUB=$ROOT/Subroutine/libstandardU.so
CPUS=${CPUS:-8}
MAXP=${MAXP:-2}
mkdir -p logs

run_one() {  # run_one <name>
    local name=$1 dir=$ROOT/data/raw/$STUDY/simulations/$1
    if [ -f "$dir/$name.odb" ] && grep -q "SUCCESSFULLY" "$dir/$name.sta" 2>/dev/null; then
        echo "  skip $name (done)"; return 0
    fi
    mkdir -p "$dir"; cd "$dir"
    cp -f "$ROOT/data/raw/$STUDY/geometry/$name.inp" .
    echo "usub_lib_dir='$(dirname "$SUB")'" > abaqus_v6.env   # Abaqus loads the compiled USDFLD from there
    abaqus job="$name" input="$name.inp" cpus=$CPUS mp_mode=mpi \
           double=both interactive > "$name.runlog" 2>&1 || true
    cd "$ROOT"
    echo "[$(date +%H:%M:%S)] $name $(grep -q SUCCESSFULLY "$dir/$name.sta" 2>/dev/null && echo OK || echo FAILED)"
}

echo "[1/5] Generating 200 input decks (abaqus cae meshing)..."
# Run as a module (-m): the generator uses relative imports
$PY -m src.preprocessing.generate_cohesive_laminate_uniform \
    --n 200 --seed 42 --layup "90" --ply-thickness 0.125 --elem-size 1.5 \
    --study $STUDY

echo "[2/5] Checking decks (keywords, thickness, BCs, output interval, mesh plot)..."
grep -q "\*Coupling" data/raw/$STUDY/geometry/plate_001.inp \
    && grep -q "\*Kinematic" data/raw/$STUDY/geometry/plate_001.inp \
    || { echo "BC MISMATCH: expected a kinematic *Coupling; aborting"; exit 1; }
# Stop on an element-size mismatch (the checker itself only warns)
CHECK_OUT=$($PY tools/check_singleply_decks.py --study $STUDY --thickness 0.125 --elem-size 1.5 --plot 4)
echo "$CHECK_OUT"
echo "$CHECK_OUT" | grep -q "median in-plane edge" \
    && { echo "MESH SIZE MISMATCH (see the warnings above); aborting"; exit 1; }

echo "[3/5] Smoke-running plate_001..."
run_one plate_001
grep -q "SUCCESSFULLY" data/raw/$STUDY/simulations/plate_001/plate_001.sta \
    || { echo "SMOKE RUN FAILED (see plate_001.runlog/.msg); aborting"; exit 1; }

echo "[4/5] Running the remaining simulations ($MAXP parallel x $CPUS cpus)..."
running=0
for inp in data/raw/$STUDY/geometry/plate_*.inp; do
    name=$(basename "$inp" .inp)
    run_one "$name" &
    running=$((running+1))
    if [ "$running" -ge "$MAXP" ]; then wait -n; running=$((running-1)); fi
done
wait
echo "  completed: $(grep -l SUCCESSFULLY data/raw/$STUDY/simulations/*/*.sta 2>/dev/null | wc -l)/200"
$PY tools/check_singleply_decks.py --study $STUDY --thickness 0.125 --elem-size 1.5 --plot 0 || true

echo "[5/5] Building dataset (+damage feature, dt=0.01)..."
$PY -m src.preprocessing.build_cohesive_dataset \
    --study $STUDY --include-damage-feature --resample-dt 0.01 --workers 8

echo "DATASET DONE:"; ls -lh data/processed/$STUDY/*.pt 2>/dev/null
