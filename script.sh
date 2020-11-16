#!/bin/zsh

python train.py --dataset=NWPU --stop_epoch=400

python train.py --dataset=UCMerced --stop_epoch=400
