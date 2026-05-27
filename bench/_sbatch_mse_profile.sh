#!/bin/bash
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --job-name=mse_profile
#SBATCH --output=_run_mse_profile_%j.out
#
# Capture real operand tensors from Llama-3.1-8B (once), then run the
# kernel-level SC MSE sweep (stoc_len + halve x SC_OWEN_MODE scramble).
set -eo pipefail
source ~/.bashrc
conda activate annstention
export HF_HUB_CACHE=/nfs/turbo/coe-nbleier/zhkangqi/hf_cache_hub
cd /scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm_llama/kernels

echo "=== host: $(hostname) ==="
PT=bench/captured_llama8b.pt
if [ ! -f "$PT" ]; then
  echo "=== capturing real operand tensors (Llama-3.1-8B, layers 0/15/31) ==="
  python -u bench/capture_real_tensors.py --layers 0,15,31 --out "$PT" \
    2>&1 | grep -vE 'Loading checkpoint shards.*it/s\]'
fi

echo "=== MSE sweep ==="
python -u bench/mse_profile.py --tensors "$PT" --out bench/mse_profile.csv
