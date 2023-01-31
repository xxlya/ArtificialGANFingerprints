#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
--data_dir './data/img_align_celeba' --use_celeba_preprocessing --image_resolution 128 \
--output_dir './results' --fingerprint_length 100 --batch_size 64 \



