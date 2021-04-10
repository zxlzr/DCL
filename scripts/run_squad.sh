
export SQUAD_DIR=SQuAD1.1
# MODEL_NAME_OR_PATH=/home/xx/pretrained_model/bert-base-uncased
MODEL_NAME_OR_PATH=./model_trained/bert_cl

# DCL file
# MODEL_NAME_OR_PATH=./model_trained/DA
# MODEL_NAME_OR_PATH=./model_trained/bert_cl
# perturb file to enhanced the ability of model

# NEW DCL FILE
# MODEL_NAME_OR_PATH=./output/DCL


# DA DA DA
TRAIN_FILE=${SQUAD_DIR}_test/train-v1.1.json
TEST_FILE=${SQUAD_DIR}_test/dev_perturb.json

TIME_STAMP=2020_9_27_19

# output_dir 用来存放checkpoints


for((i=3;i<=7;i++));
do
CUDA_VISIBLE_DEVICES=1 python run_squad.py \
  --model_type bert \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --do_eval \
  --eval_all_checkpoints \
  --do_lower_case \
  --train_file $TRAIN_FILE \
  --predict_file $TEST_FILE \
  --per_gpu_train_batch_size 12 \
  --learning_rate ${i}e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./output/DCL_DA_lr${i}${TIME_STAMP} \
  --data_dir ./dataset \
  --save_steps 1000
done
