#!/bin/bash
python save_features.py --method=protonet --dataset=WURS46 --n_shot=5
python test_s1.py --method=protonet --dataset=WURS46 --n_shot=5