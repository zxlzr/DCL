CUDA_VISIBLE_DEVICES=2 python main.py --gpus="0," --max_epochs=300  --num_workers=8 \
    --model_name_or_path ./origin_bert \
    --accumulate_grad_batches 3 \
    --batch_size 8 \
    --data_dir dataset/dialogue_8shot \
    --check_val_every_n_epoch 3 \
    --model_class BertForSequenceClassification \
    --wandb  \
    --lr 3e-5
