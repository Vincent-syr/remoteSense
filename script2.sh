#!/bin/bash
# python train.py --method=protonet --dataset=CUB --n_shot=5 --stop_epoch=300
# python save_features.py --method=protonet --dataset=CUB --n_shot=5
# python test_s1.py --method=protonet --dataset=CUB --n_shot=5


# 21.04.19   
# 11:15
python train.py --method=protonet --dataset=NWPU --n_shot=5 --stop_epoch=300
python train.py --method=protonet --dataset=NWPU --n_shot=5 --no_aug --stop_epoch=300 
# 21:01
python 
python train.py --method=cs_protonet --dataset=NWPU --n_shot=5 --stop_epoch=300

# 21.04.20
# 9:12 without cs_mlp  tmux 3-0
python train.py --method=cs_protonet --dataset=NWPU --n_shot=5 --stop_epoch=300
# 17:36  tmux 1-0 
python train.py --method=protonet --dataset=NWPU --n_shot=1 --stop_epoch=300   
# 19:37 tmux 1-2    2.9h
python train.py --method=cs_protonet --dataset=NWPU --n_shot=1 --stop_epoch=300   
# 22:59 tmux 1-2
python train.py --method=s2m2_cs --dataset=NWPU --n_shot=5 --stop_epoch=300 
# 23:01 tmux 1-0  
python train.py --method=s2m2_cs --dataset=NWPU --n_shot=1 --stop_epoch=300   


# 04/21
# 15点 hz1 tmux1-0
python save_features.py --method=s2m2_cs --dataset=NWPU --n_shot=5
python test_s1.py --method=s2m2_cs --dataset=NWPU --n_shot=5
python save_features.py --method=s2m2_cs --dataset=NWPU --n_shot=1
python test_s1.py --method=s2m2_cs --dataset=NWPU --n_shot=1
# 15:20 hz1 tmux1-0
python save_features.py --method=cs_protonet --dataset=NWPU --n_shot=5
python test_s1.py --method=cs_protonet --dataset=NWPU --n_shot=5
python save_features.py --method=cs_protonet --dataset=NWPU --n_shot=1
python test_s1.py --method=cs_protonet --dataset=NWPU --n_shot=1
#15:34 hz1 tmux1-2 device0
python train.py --method=s2m2_cs --dataset=NWPU --n_shot=5 --stop_epoch=300 
# 15:37 hz1 tmux 1-0 device1
python train.py --method=s2m2_cs --dataset=NWPU --n_shot=1 --stop_epoch=300   

# 17:30 czy1 tmux1-0
CUDA_VISIBLE_DEVICES=0 python train.py --method=cs_protonet --dataset=NWPU --n_shot=5 --stop_epoch=300 
# 17:30 czy1 tmux1-1
CUDA_VISIBLE_DEVICES=1 python train.py --method=cs_protonet --dataset=NWPU --n_shot=1 --stop_epoch=300 
# 17:30 czy1 tmux1-2
CUDA_VISIBLE_DEVICES=1 python train.py --method=protonet --dataset=NWPU --n_shot=1 --stop_epoch=300

# 19:40 hz1 tmux 1-0    4.6h
CUDA_VISIBLE_DEVICES=1 python train.py --method=protonet --dataset=NWPU --n_shot=5 --stop_epoch=300   
# 20:44
python save_features.py --method=s2m2_cs --dataset=NWPU --n_shot=5
python test_s1.py --method=s2m2_cs --dataset=NWPU --n_shot=5
python save_features.py --method=s2m2_cs --dataset=NWPU --n_shot=1
python test_s1.py --method=s2m2_cs --dataset=NWPU --n_shot=1
# 21:25
python save_features.py --method=cs_protonet --dataset=NWPU --n_shot=1
python test_s1.py --method=cs_protonet --dataset=NWPU --n_shot=1
python save_features.py --method=protonet --dataset=NWPU --n_shot=1
python test_s1.py --method=protonet --dataset=NWPU --n_shot=1

python save_features.py --method=cs_protonet --dataset=NWPU --n_shot=5
python test_s1.py --method=cs_protonet --dataset=NWPU --n_shot=5

# 04/22
python save_features.py --method=protonet --dataset=NWPU --n_shot=5
python test_s1.py --method=protonet --dataset=NWPU --n_shot=


# 04/23
#16:00  hz1 tmux1-0 device0
CUDA_VISIBLE_DEVICES=0 python train.py --method=s2m2_cs --dataset=WURS46 --n_shot=5 --stop_epoch=300 
# 18:30 hz1 tmux1-2
CUDA_VISIBLE_DEVICES=1 python train.py --method=cs_protonet --dataset=WURS46 --n_shot=5 --stop_epoch=300 
# 18:30 czy1 tmux3-0
CUDA_VISIBLE_DEVICES=0 python train.py --method=protonet --dataset=WURS46 --n_shot=5 --stop_epoch=300 
# 19:22 czy1 tmux3-1
CUDA_VISIBLE_DEVICES=1 python train.py --method=protonet --dataset=WURS46 --n_shot=1 --stop_epoch=500 
# 20:22 czy1 tmux3-2
CUDA_VISIBLE_DEVICES=1 python train.py --method=cs_protonet --dataset=WURS46 --n_shot=1 --stop_epoch=500 

python save_features.py --method=s2m2_cs --dataset=WURS46 --n_shot=5
python test_s1.py --method=s2m2_cs --dataset=WURS46 --n_shot=5

# 4/24
python save_features.py --method=cs_protonet --dataset=WURS46 --n_shot=5
python test_s1.py --method=cs_protonet --dataset=WURS46 --n_shot=5
# 23:49 hz1 tmux1-2
CUDA_VISIBLE_DEVICES=1 python train.py --method=s2m2_cs --dataset=WURS46 --n_shot=1 --stop_epoch=500 
