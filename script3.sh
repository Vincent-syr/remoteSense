#!/bin/bash
# python train.py --method=rotate --dataset=OPTIMAL --n_shot=5 --stop_epoch=300 --no_aug
# python save_features.py --method=rotate --dataset=OPTIMAL --n_shot=5 --no_aug
# python test_s1.py --method=rotate --dataset=OPTIMAL --n_shot=5  --no_aug

python train.py --method=rotate --dataset=OPTIMAL --n_shot=5 --stop_epoch=300 --no_aug
python save_features.py --method=protonet --dataset=OPTIMAL --n_shot=5 --stop_epoch=300 
python test_s1.py --method=protonet --dataset=OPTIMAL --n_shot=5 

# python save_features.py --method=protonet --dataset=CUB --n_shot=5 
# python test_s1.py --method=protonet --dataset=CUB --n_shot=5 