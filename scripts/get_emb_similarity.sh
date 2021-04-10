#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
TASK_NAME=SST-2
#TASK_NAMES=(CoLA RTE STS-B)
#TASK_NAMES=(MRPC)
MODEL_TYPE=bert-base-uncased
NORM_TYPES=(layer unnorm batch power)
BATCH_SIZE=192
LR=2e-5

for norm_type in ${NORM_TYPES[@]}
do
  echo "Starting contrast bert on da $TASK_NAME:"
  OUT_DIR=new_result/bert_${norm_type}_pretrain_glue/$TASK_NAME
  LOGGING=runs/bert_${norm_type}_pretrain/$TASK_NAME
  python similarity_contrast.py  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --data_dir glue_data/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size=$BATCH_SIZE \
  --num_train_epochs 3 \
  --output_dir $OUT_DIR \
  --mlm \
  --learning_rate $LR \
  --logging_dir  $LOGGING \
  --weight_decay=0.1 \
  --overwrite_output_dir \
  --norm_type=$norm_type
done

