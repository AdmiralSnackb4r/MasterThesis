#!/bin/bash
#SBATCH --account=hai_1008
#SBATCH --mail-user=e.kromm@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --nodes=2
#SBATCH --job-name=trainRelTRpre
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --output=output.%j
#SBATCH --error=error.%j
#SBATCH --time=24:00:00
#SBATCH --partition=booster

# Get number of cpu per task
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Extracts the first hostname from the list of allocated nodes to use as the master address.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Modifies the master address to allow communication over InfiniBand cells.
MASTER_ADDR="${MASTER_ADDR}i"
# Get IP for hostname.
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=7010

cd $HOME/master/
source sc_venv_template/activate.sh

srun --cpu_bind=none bash -c "torchrun_jsc \
   --nnodes=$SLURM_NNODES \
   --rdzv_backend c10d \
   --nproc_per_node=gpu \
   --rdzv_id $RANDOM \
   --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
   --rdzv_conf=is_host=\$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi) \
    RelTR/pretrain_reltr.py "

# srun --cpu_bind=none python TrainOnCityScapes/scripts/TrainResnetDistributed.py