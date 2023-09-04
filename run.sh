#!/bin/bash

SCRIPT_DIR=$PBS_O_WORKDIR

export TORCH_EXTENSIONS_DIR="${SCRIPT_DIR}/.project_cache/torch-extensions/"
export TUNE_RESULT_DIR="/scratch/taw2/.cache/raytune/"
export HF_HOME="/scratch/taw2/.cache/huggingface/"
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

# For future use. Gets master ip address
master_ip=`hostname -I | awk '{print $1}'`
RDZV_ENDPOINT=$1
USER=$2
ENV_PATH=$3
NNODES=$4
NGPUS=$5

echo "==============================" >> $HOSTNAME.txt

# Causes exit from script on failure
set -e

# Add modules (implementation dependent, may not be required)
echo ""
echo "***** Job initialized, loading modules... *****"
echo ""

# Source file for setting up modules
source /etc/profile.d/modules.sh

module add cuda/11.6.2-gcc/9.5.0
module add nccl/2.11.4-1-gcc/9.5.0-cu11_6-nvP-nvV-nvA
module add openmpi/4.1.3-gcc/9.5.0-cu11_6-nvP-nvV-nvA-ucx
module add anaconda3/2022.05-gcc/9.5.0

echo ""
echo "***** Navigating to directory... *****"
echo ""
cd $SCRIPT_DIR
pwd

echo "***** Activating environment... *****"
source activate $ENV_PATH

export WANDB_WATCH="all"
# export WANDB_API_KEY=""

torchrun \
	--nnodes=$NNODES \
	--nproc_per_node=$NGPUS \
	--rdzv_id=12345 \
	--rdzv_backend=c10d \
	--rdzv_endpoint=$RDZV_ENDPOINT:3008 \
	$6

echo "$HOSTNAME" Finished Tasks
# echo ""
# echo "Benchmarking..."
# echo ""
# python3 scripts/benchmarking.py
