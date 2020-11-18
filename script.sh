#!/bin/zsh

python train.py --dataset=NWPU --stop_epoch=400

python train.py --dataset=UCMerced --stop_epoch=400
# python train.py --dataset=OPTIMAL --stop_epoch=400
# python train.py --dataset=WURS --stop_epoch=400
# python train.py --dataset=AIDS --stop_epoch=300    11/17/16:38
# python save_features.py --dataset=CUB --method=rotate
# python test_s1.py --dataset=CUB --method=rotate

# python train.py --method=rotate --dataset=CUB   
# python train.py --method=rotate --dataset=CUB  --start_epoch=299 --stop_epoch=500
# python train.py --method=rotate --dataset=NWPU    11/18 13:33, 12.8h
