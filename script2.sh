#!/bin/bash
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