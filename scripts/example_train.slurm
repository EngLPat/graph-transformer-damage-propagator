#!/bin/bash
#SBATCH --job-name=gtp_train
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Example SLURM job: train the laminate model with the hyperparameters of the
# paper on two GPUs. Replace the <placeholders> and submit from the repository
# root.
#
# batch_size is per rank, so 2 ranks x 8 give an effective batch of 16.
# GTP_MAX_TRAIN_SECONDS stops training cleanly before the wall-time limit, and
# resubmitting the job resumes from best_models/last_checkpoint.pt.

cd "$SLURM_SUBMIT_DIR"

# Adjust to your conda installation
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gtp

# After conda activation, whose hooks may return a non-zero status
set -eo pipefail

export OMP_NUM_THREADS=1
export GTP_MAX_TRAIN_SECONDS=84600
export GTP_STAMPED_CKPTS=1        # keep a checkpoint every 10 epochs

DATASET=data/processed/laminate/laminate_dmgstate_dataset_uniform_dt0.01.pt
OUTDIR=results/laminate_reference

mkdir -p logs "$OUTDIR"

RESUME_ARGS=""
if [ -f "$OUTDIR/best_models/last_checkpoint.pt" ]; then
  RESUME_ARGS="--resume-from $OUTDIR/best_models/last_checkpoint.pt"
fi

torchrun --nproc_per_node=2 --standalone -m src.gtp \
  --mode train \
  --dataset "$DATASET" \
  --num-layers 9 --hidden-dim 128 --attention-freq 4 --dropout 0.00164 \
  --batch-size 8 --learning-rate 0.00019068 --weight-decay 2.5e-5 \
  --rate-cap 0 --epochs 250 --two-stage \
  --seed 42 \
  $RESUME_ARGS \
  --output-dir "$OUTDIR"
