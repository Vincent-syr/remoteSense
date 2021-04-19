#!/bin/bash
# python train.py --method=protonet --dataset=CUB --n_shot=5 --stop_epoch=300
# python save_features.py --method=protonet --dataset=CUB --n_shot=5
# python test_s1.py --method=protonet --dataset=CUB --n_shot=5


# 21.04.19   
# 11:15
python train.py --method=protonet --dataset=NWPU --n_shot=5 --stop_epoch=300
python train.py --method=protonet --dataset=NWPU --n_shot=5 --train_aug=False --stop_epoch=300 

# 20:18
python 
python train.py --method=cs_protonet --dataset=NWPU --n_shot=5 --stop_epoch=300