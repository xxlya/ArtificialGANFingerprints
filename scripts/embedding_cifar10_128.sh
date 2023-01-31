#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python embed_fingerprints.py \
--encoder_path './results/CIFAR10_32x32_encoder.pth' --decoder_path './results/CIFAR10_32x32_decoder.pth' \
--data_dir  './data/cifar10' \
--image_resolution 32 \
--output_dir './results' \
--identical_fingerprints \
--batch_size 64 \
--dataset cifar10  --check\
--identical_fingerprints \

