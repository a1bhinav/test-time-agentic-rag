#!/bin/bash
#SBATCH -p gpu-preempt # Partition
#SBATCH -G 2    # Number of GPUs
#SBATCH -c 32   # Number of CPU cores
#SBATCH --mem=100GB # Requested Memory
#SBATCH -N 1
#SBATCH -t 0-04:00:00 # 4 hrs
#SBATCH -o output_inference-%j.out # Specify where to save terminal output, %j = job ID will be filled by slurm
#SBATCH --constraint=vram48

cd /work/pi_wenlongzhao_umass_edu/2/dguntur/Search-R1

# define my own environment variables for huggingface cache files (prevent locks)
export HF_HOME=/home/zzuo_umass_edu/hf_home_link
export HF_HUB_CACHE=$HF_HOME

module load conda/latest
conda activate /project/pi_wenlongzhao_umass_edu/2/dguntur/.conda/envs/retriever
bash retrieval_launch.sh > /dev/null &

echo "Waiting for retrieval server to return 200..."

until [ "$(curl -s -o /dev/null -w "%{http_code}" -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"queries": ["What is the capital of France?"], "topk": 3, "return_scores": true}')" == "200" ]; do
  echo "Server not ready (not returning 200). Retrying in 1 minute..."
  sleep 60
done

echo "Server is returning 200! Starting training..."


# conda activate searchr1
# bash scripts/nq_hotpotqa/v0.2/train_ppo.sh

source /home/zzuo_umass_edu/groupdir/dguntur/Search-R1/.searchr1/bin/activate
python3 infer_hotpot_500_zhiyang.py
