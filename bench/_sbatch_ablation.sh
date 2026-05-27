#!/bin/bash
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --job-name=sc_ablation
#SBATCH --output=_run_ablation_%j.out
set -eo pipefail
source ~/.bashrc
conda activate annstention
export HF_HUB_CACHE=/nfs/turbo/coe-nbleier/zhkangqi/hf_cache_hub
cd /scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm_llama/kernels
echo "=== host: $(hostname) ==="
PT=bench/captured_llama8b.pt
[ -f "$PT" ] || python -u bench/capture_real_tensors.py --layers 0,15,31 --out "$PT" 2>&1 | grep -vE 'it/s\]'
python -u bench/ablation.py --tensors "$PT" --out bench/ablation.csv
