#!/bin/zsh

# python train.py --dataset=NWPU --stop_epoch=400

# python train.py --dataset=UCMerced --stop_epoch=400
# python train.py --dataset=OPTIMAL --stop_epoch=400
# python train.py --dataset=WURS --stop_epoch=400
# python train.py --dataset=AIDS --stop_epoch=300    11/17/16:38
# python save_features.py --dataset=CUB --method=rotate
# python test_s1.py --dataset=CUB --method=rotate

# python train.py --method=rotate --dataset=CUB   
# python train.py --method=rotate --dataset=CUB  --start_epoch=299 --stop_epoch=500
# python train.py --method=rotate --dataset=NWPU    11/18 13:33, 12.8h

# 12/22
# python train.py --method=rotate --dataset=UCMerced    

# 11:30 lab1
# python train.py --method=protonet --dataset=AIDS --n_shot=1 --stop_epoch=400    # ETA: 12.3h
# python train.py --method=protonet --dataset=OPTIMAL --n_shot=1 --stop_epoch=400  # ETA: 5.6h

# 19:44 huang_sec
python train.py --method=protonet --dataset=NWPU --n_shot=1 --stop_epoch=400   # 7.3h t0
python train.py --method=protonet --dataset=WURS --n_shot=1 --stop_epoch=400   # 8.7h t1
# 23:14 huang_2 rotate 1 shot
python train.py --method=rotate --dataset=OPTIMAL --n_shot=1 --stop_epoch=300  #  t2
# huang_3
python train.py --method=rotate --dataset=WURS --n_shot=5 --stop_epoch=300  #  t2

# 12/23
# 15:18 huang_2
python train.py --method=rotate --dataset=OPTIMAL --n_shot=5 --stop_epoch=300 
# 15:20 huang_3
python train.py --method=rotate --dataset=WURS --n_shot=1 --stop_epoch=300