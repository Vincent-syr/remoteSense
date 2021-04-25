Use few shot learning algorithm to solve problem in Remote Sense Scene Classification
env：
    pytorch1.4.0
    torchvision=0.5.0
    cudatoolkit=10.0

train:
python train.py --method=s2m2_cs --dataset=NWPU --n_shot=5 --stop_epoch=300 
save feature:
python save_features.py --method=s2m2_cs --dataset=NWPU --n_shot=5
test:
python test_s1.py --method=s2m2_cs --dataset=NWPU --n_shot=5

指定device：
CUDA_VISIBLE_DEVICES=0 python train.py