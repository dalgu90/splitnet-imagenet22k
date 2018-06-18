#!/bin/sh
export CUDA_VISIBLE_DEVICES=1,2,3,4,5
export LD_PRELOAD="/usr/lib/libtcmalloc.so"
train_dir="./group_split-1-4-4"
train_dataset="scripts/22k_train.txt"
train_image_root="/data1/dalgu/imagenet22k_resized/imagenet22k_resized/"
val_dataset="scripts/22k_test.txt"
val_image_root="/data1/dalgu/imagenet22k_resized/imagenet22k_resized/"

python train.py --train_dir $train_dir \
    --train_dataset $train_dataset \
    --train_image_root $train_image_root \
    --val_dataset $val_dataset \
    --val_image_root $val_image_root \
    --num_gpus 4 \
    --batch_size 64 \
    --val_interval 1000 \
    --val_iter 100 \
    --ngroups1 4 \
    --ngroups2 4 \
    --ngroups3 1 \
    --l2_weight 0.0001 \
    --gamma1 1.0 \
    --gamma2 1.0 \
    --gamma3 3.0 \
    --initial_lr 0.001 \
    --lr_step_epoch 10.0,20.0 \
    --lr_decay 0.1 \
    --bn_no_scale True \
    --weighted_group_loss True \
    --max_steps 831000 \
    --checkpoint_interval 13850 \
    --group_summary_interval 2770 \
    --gpu_fraction 0.96 \
    --display 100 \
    --finetune True \
    --basemodel "./resnet_baseline2/model.ckpt-138729" \
    #--checkpoint "./group_split-1-2-2/model.ckpt-150000" \
    #--load_last_layers True \

# SplitNet of ResNet-50x2 on ImageNet 22k
# Reduced gradients
# Finetune with basemodel: all layers
