#!/bin/bash
# python save_features.py --method=s2m2_cs --dataset=NWPU --n_shot=1 --lr_anneal=pwc
# python test_s1.py --method=s2m2_cs --dataset=NWPU --n_shot=1 --lr_anneal=pwc

# CUDA_VISIBLE_DEVICES=1 python save_features.py --method=cs_protonet --dataset=NWPU --n_shot=1 --lr_anneal=pwc
# CUDA_VISIBLE_DEVICES=1 python test_s1.py --method=cs_protonet --dataset=NWPU --n_shot=1 --lr_anneal=pwc

CUDA_VISIBLE_DEVICES=1 python save_features.py --method=protonet --dataset=NWPU --n_shot=1 --lr_anneal=pwc
CUDA_VISIBLE_DEVICES=1 python test_s1.py --method=protonet --dataset=NWPU --n_shot=1 --lr_anneal=pwc