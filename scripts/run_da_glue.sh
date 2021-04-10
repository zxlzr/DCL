#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
GLUE_DIR=glue_data
#TASK_NAMES=(SST-2, STS-B)
TASK_NAMES=(QNLI)
MODEL_TYPE=bert-base-uncased
#LR=2e-5
BATCH_SIZE=48
for TASK_NAME in ${TASK_NAMES[@]}
do
  echo "Starting DCL finetune bert on $TASK_NAME:"
  for i in 2 3 4 5
  do
    echo "    Inside loop LR: $i"
    OUT_DIR=glue_result/bert_da_finetune/${TASK_NAME}/lr_${i}
    LOGGING=runs/bert_da_finetune/$TASK_NAME/lr_${i}
    python run_glue_from_da.py --model_name_or_path  $MODEL_TYPE \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_device_eval_batch_size=$BATCH_SIZE \
    --per_device_train_batch_size=$BATCH_SIZE \
    --learning_rate ${i}e-5   \
    --num_train_epochs 4 \
    --output_dir $OUT_DIR \
    --evaluate_during_training \
    --logging_dir $LOGGING \
    --overwrite_output_dir
    done
done

