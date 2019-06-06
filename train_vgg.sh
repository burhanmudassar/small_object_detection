#!/bin/bash
while true; do
     python3 train.py --cfg=experiments/vgg16/ssd_dual_vgg16_train_coco_1.yml --multi_gpu
done
