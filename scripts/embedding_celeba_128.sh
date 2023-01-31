#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python embed_fingerprints.py \
--encoder_path './results/CelebA_128x128_encoder.pth' --decoder_path './results/CelebA_128x128_decoder.pth' \
--data_dir  './data/img_align_celeba' \
--image_resolution 128 \
--output_dir './fig' \
--identical_fingerprints \
--batch_size 64 \
--dataset celeba  --check \
--identical_fingerprints \
