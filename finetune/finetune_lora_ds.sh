#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,2,3,4

DIR=`pwd`

GPUS_PER_NODE=4
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=9911 # 6001->9911

DIR_ID=$(date '+%Y%m%d-%H%M%S')

#TODO: 1 set your model path, dataset path and output path
MODEL="/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/Qwen-VL-Chat" #"Qwen/Qwen-VL-Chat"/"Qwen/Qwen-VL" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
#DATA="/data1/llm/houzm/99-code/01-Qwen-VL/ai_doctor/data/data_finetune/align/align_finetune_dataset.json" # alignment dataset
#OUTPUT_DIR="/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/Qwen-VL-Chat/hzm_qwen_finetune/align/$(date '+%Y%m%d-%H%M%S')" # alignment model output path

#
#MODEL="/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/Qwen-VL-Chat/hzm_qwen_finetune/align/20240721-211309"

OUTPUT_DIR="/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/Qwen-VL-Chat/hzm_qwen_finetune/diagnose/$DIR_ID" # diagnose model output path
DATA="$OUTPUT_DIR/train_test_dataset/diagnose_finetune_dataset.json" # diagnose dataset

#TODO: 为什么指定GPU多卡训练，这两个参数传入无法正常运行该sh文件？-- 待排查原因
#    --nnodes $NNODES \
#    --node_rank $NODE_RANK \
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
# TODO: modify settings
# model_max_length 2048->4096
# report_to "none" -> tensorboard

# TODO: fine tuning the following parameters
# num_train_epochs 5->2
# per_device_train_batch_size 1
# per_device_eval_batch_size 1
# gradient_accumulation_steps 16->1
# learning_rate 1e-6->1e-5

run_sh="/data1/llm/anaconda3/envs/hzm-qwen-vl/bin/torchrun  $DISTRIBUTED_ARGS /data1/llm/houzm/99-code/01-Qwen-VL/finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 True \
    --fix_vit True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
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
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed /data1/llm/houzm/99-code/01-Qwen-VL/finetune/ds_config_zero3.json\
    --use_lora \
    "

mkdir -p $OUTPUT_DIR

echo $OUTPUT_DIR

# 1 生成 train and test dataset
/data1/llm/anaconda3/envs/hzm-qwen-vl/bin/python3.8 /data1/llm/houzm/99-code/01-Qwen-VL/ai_doctor/data/data_preprocess/xlsx_2_json.py --task SC  2>&1 | tee "$OUTPUT_DIR/train_model.log"
# 2 拷贝 dataset 到 训练脚本 目录下
mkdir -p $OUTPUT_DIR/train_test_dataset
cp -r /data1/llm/houzm/99-code/01-Qwen-VL/ai_doctor/data/data_finetune/diagnose/*.json $OUTPUT_DIR/train_test_dataset
# 3 训练 模型
eval $run_sh 2>&1 | tee "$OUTPUT_DIR/train_model.log"
# 4 执行 预测
/data1/llm/anaconda3/envs/hzm-qwen-vl/bin/python3.8 /data1/llm/houzm/99-code/01-Qwen-VL/ai_doctor/test/qwen_lora_diagnose_test.py --dir-id $DIR_ID  2>&1 | tee "$OUTPUT_DIR/train_model.log"
