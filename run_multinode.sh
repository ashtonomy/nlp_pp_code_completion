#PBS -N multi_run
#PBS -l select=2:ncpus=40:ngpus=2:gpu_model=a100:mem=245gb:phase=29,walltime=72:00:00
#PBS -j oe
#PBS -o logs/multi_run.log

SCRIPT_DIR=$PBS_O_WORKDIR
model_name=codellama/CodeLlama-13b-hf

export HF_HOME="/scratch/taw2/.cache/huggingface/"
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1 

# Add modules (implementation dependent, may not be required)
echo ""
echo "***** Job initialized, loading modules... *****"
echo ""
module add cuda/11.6.2-gcc/9.5.0
module add nccl/2.11.4-1-gcc/9.5.0-cu11_6-nvP-nvV-nvA
module add openmpi/4.1.3-gcc/9.5.0-cu11_6-nvP-nvV-nvA-ucx
module add anaconda3/2022.05-gcc/9.5.0

echo ""
echo "***** Navigating to directory... *****"
echo ""

cd $SCRIPT_DIR 
pwd

echo "***** Activating environment *****"
env_name="hf_env"
source activate $env_name

nnodes=$(cat $PBS_NODEFILE | wc -l)
ncpus=$NCPUS
ngpus=2
nprocs=$((ncpus / 5))

timestamp=$(date +%D_%H_%M_%S | tr / _)
OUTPUT_DIR="${PBS_O_WORKDIR/home/scratch}/output/nlp_pp_${model_name////_}_${timestamp}"
mkdir -p $OUTPUT_DIR

echo "Running as ${USER} with ${nnodes} nodes."

export TORCH_EXTENSIONS_DIR="${pwd}/.project_cache/torch-extensions/"

pbsdsh -- bash "$(pwd)"/run.sh $HOSTNAME $USER $env_name $nnodes $ngpus "run_clm.py \
        --deepspeed ${SCRIPT_DIR}/deepspeed_configs/llama_z3_offload.json \
        --model_name_or_path $model_name \
        --output_dir ${OUTPUT_DIR} \
	--ddp_timeout 9000 \
	--dataset_name AshtonIsNotHere/nlp_pp_code_dataset \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
	--do_train \
        --do_eval \
        --learning_rate 2e-5 \
        --weight_decay 0.1 \
        --gradient_accumulation_steps 8 \
        --warmup_ratio 0.1 \
        --evaluation_strategy epoch \
        --save_strategy epoch \
	--load_best_model_at_end \
        --report_to wandb \
	--torch_dtype bfloat16 \
	--num_train_epochs 5 \
	--block_size 1024 \
	--gradient_checkpointing \
	--run_name z3_${model_name////_}"

# --preprocessing_num_workers $nprocs \
