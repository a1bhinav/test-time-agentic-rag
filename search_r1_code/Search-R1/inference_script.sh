#!/bin/bash
#SBATCH -p gpu-preempt # Partition
#SBATCH -G 2    # Number of GPUs
#SBATCH -c 32   # Number of CPU cores
#SBATCH --mem=100GB # Requested Memory
#SBATCH -N 1
#SBATCH -t 0-04:00:00 # 4 hrs
#SBATCH -o output_inference-%j.out # Specify where to save terminal output, %j = job ID will be filled by slurm
#SBATCH --constraint=vram48

# Run from the Search-R1 directory (this script's location).
cd "$(dirname "$(readlink -f "$0")")"

# Optional: redirect HuggingFace caches to a writable location to avoid lock files
# on shared filesystems. Leave unset to use the default ~/.cache/huggingface.
# export HF_HOME=./hf_cache
# export HF_HUB_CACHE=$HF_HOME

# Activate your environment. Uncomment and edit ONE of the lines below to match
# your local setup.
# module load conda/latest && conda activate <YOUR_CONDA_ENV>
# source <YOUR_VENV>/bin/activate

bash retrieval_launch.sh > /dev/null &

echo "Waiting for retrieval server to return 200..."

until [ "$(curl -s -o /dev/null -w "%{http_code}" -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"queries": ["What is the capital of France?"], "topk": 3, "return_scores": true}')" == "200" ]; do
  echo "Server not ready (not returning 200). Retrying in 1 minute..."
  sleep 60
done

echo "Server is returning 200! Starting inference..."

# Run one of the test-time strategy scripts. Uncomment the variant you want.
python3 infer_hotpot_500_no_dup_docs.py
# python3 infer_hotpot_500_caching.py
