#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

DIR=`pwd`

GPUS_PER_NODE=5
NNODES=5
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=9911 # 6001->9911

#TODO: 1 set your model path, dataset path and output path
#MODEL="/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/Qwen-VL-Chat" #"Qwen/Qwen-VL-Chat"/"Qwen/Qwen-VL" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
#DATA="/data1/llm/houzm/99-code/01-Qwen-VL/ai-doctor/data/data_finetune/align/align_finetune_dataset.json" # alignment dataset
#OUTPUT_DIR="/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/Qwen-VL-Chat/hzm_qwen_finetune/align/$(date '+%Y%m%d-%H%M%S')" # alignment model output path

MODEL="/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/Qwen-VL-Chat/hzm_qwen_finetune/align/20240721-211309"
DATA="/data1/llm/houzm/99-code/01-Qwen-VL/ai-doctor/data/data_finetune/diagnose/diagnose_finetune_dataset.json" # diagnose dataset
OUTPUT_DIR="/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/Qwen-VL-Chat/hzm_qwen_finetune/diagnose/$(date '+%Y%m%d-%H%M%S')" # diagnose model output path

#TODO: 为什么指定GPU多卡训练，这两个参数传入无法正常运行该sh文件？-- 待排查原因
#    --nnodes $NNODES \
#    --node_rank $NODE_RANK \
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
# TODO: modify settings
# model_max_length 2048->4096
# report_to "none" -> tensorboard
run_sh="/data1/llm/anaconda3/envs/hzm-qwen-vl/bin/torchrun  $DISTRIBUTED_ARGS /data1/llm/houzm/99-code/01-Qwen-VL/finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 True \
    --fix_vit True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to tensorboard \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed /data1/llm/houzm/99-code/01-Qwen-VL/finetune/ds_config_zero3.json"

mkdir -p $OUTPUT_DIR
#cp -r /data1/llm/houzm/99-code/01-Qwen-VL/finetune.py $OUTPUT_DIR
eval $run_sh 2>&1 | tee "$OUTPUT_DIR/align_train.log"
