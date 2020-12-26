#!/bin/bash
# python train.py --method=rotate --dataset=OPTIMAL --n_shot=5 --stop_epoch=300 --train_aug=False
# python save_features.py --method=protonet --dataset=OPTIMAL --n_shot=5 --train_aug=False
# python test_s1.py --method=protonet --dataset=OPTIMAL --n_shot=5  --train_aug=False


python save_features.py --method=protonet --dataset=UCMerced --n_shot=1
python test_s1.py --method=protonet --dataset=UCMerced --n_shot=1
