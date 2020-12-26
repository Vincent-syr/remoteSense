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

# 12/24
# 8:54 huang_2
python train.py --method=rotate --dataset=UCMerced --n_shot=1 --stop_epoch=300  # 8.7h
# 8:56 huang_3
python train.py --method=rotate --dataset=NWPU --n_shot=1 --stop_epoch=300 # 9.5h
# 19:45 huang_2
python train.py --method=rotate --dataset=AIDS --n_shot=1 --stop_epoch=300
# 19:50 huang_3
python train.py --method=rotate --dataset=NWPU --n_shot=1 --start_epoch=299 --stop_epoch=500


# 12/26
# 20:50


# 21:20 rotate save feature and test, n_shot=5 
python save_features.py --method=rotate --dataset=NWPU --n_shot=5
python save_features.py --method=rotate --dataset=AIDS --n_shot=5
python save_features.py --method=rotate --dataset=UCMerced --n_shot=5
python save_features.py --method=rotate --dataset=WURS --n_shot=5
python save_features.py --method=rotate --dataset=OPTIMAL --n_shot=5
python test_s1.py --method=rotate --dataset=NWPU --n_shot=5
python test_s1.py --method=rotate --dataset=AIDS --n_shot=5
python test_s1.py --method=rotate --dataset=UCMerced --n_shot=5
python test_s1.py --method=rotate --dataset=WURS --n_shot=5
python test_s1.py --method=rotate --dataset=OPTIMAL --n_shot=5

#21:40 rotate, save feature and test, n_shot=1
python save_features.py --method=rotate --dataset=NWPU --n_shot=1
python save_features.py --method=rotate --dataset=AIDS --n_shot=1
python save_features.py --method=rotate --dataset=UCMerced --n_shot=1
python save_features.py --method=rotate --dataset=WURS --n_shot=1
python save_features.py --method=rotate --datas et=OPTIMAL --n_shot=1

python test_s1.py --method=rotate --dataset=NWPU --n_shot=1
python test_s1.py --method=rotate --dataset=AIDS --n_shot=1
python test_s1.py --method=rotate --dataset=UCMerced --n_shot=1
python test_s1.py --method=rotate --dataset=WURS --n_shot=1
python test_s1.py --method=rotate --dataset=OPTIMAL --n_shot=1

# 21:50 protonet, save feature and test, n_shot=5
python save_features.py --method=protonet --dataset=NWPU --n_shot=5
python save_features.py --method=protonet --dataset=AIDS --n_shot=5
python save_features.py --method=protonet --dataset=UCMerced --n_shot=5
python save_features.py --method=protonet --dataset=WURS --n_shot=5
python save_features.py --method=protonet --dataset=OPTIMAL --n_shot=5
python test_s1.py --method=protonet --dataset=NWPU --n_shot=5
python test_s1.py --method=protonet --dataset=AIDS --n_shot=5
python test_s1.py --method=protonet --dataset=UCMerced --n_shot=5
python test_s1.py --method=protonet --dataset=WURS --n_shot=5
python test_s1.py --method=protonet --dataset=OPTIMAL --n_shot=5

# 21:55 protonet, protonet, save feature and test, n_shot=1
python save_features.py --method=protonet --dataset=NWPU --n_shot=1
python save_features.py --method=protonet --dataset=AIDS --n_shot=1
python save_features.py --method=protonet --dataset=UCMerced --n_shot=1
python save_features.py --method=protonet --dataset=WURS --n_shot=1
python save_features.py --method=protonet --dataset=OPTIMAL --n_shot=1
python test_s1.py --method=protonet --dataset=NWPU --n_shot=1
python test_s1.py --method=protonet --dataset=AIDS --n_shot=1
python test_s1.py --method=protonet --dataset=UCMerced --n_shot=1
python test_s1.py --method=protonet --dataset=WURS --n_shot=1
python test_s1.py --method=protonet --dataset=OPTIMAL --n_shot=1


# 22:00  hz_2
python train.py --method=protonet --dataset=AIDS --n_shot=5 --stop_epoch=300
python save_features.py --method=protonet --dataset=AIDS --n_shot=5
python test_s1.py --method=protonet --dataset=AIDS --n_shot=5

# 22:20  hz_3
python train.py --method=rotate --dataset=OPTIMAL --n_shot=5 --stop_epoch=300 --train_aug=False
python save_features.py --method=protonet --dataset=OPTIMAL --n_shot=5 --train_aug=False
python test_s1.py --method=protonet --dataset=OPTIMAL --n_shot=5  --train_aug=False