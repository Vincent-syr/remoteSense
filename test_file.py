import os
import torch
checkpoint_dir = "checkpoints/WURS46/ResNet12_s2m2_cs_aug_Adam_lr0.001_const_wd0_5way_5shot"
trlog_file = os.path.join(checkpoint_dir, 'trlog', "20210423-033657")
trlog = torch.load(trlog_file)
start_epoch = 100
print(trlog.keys())
for key, val in trlog.items():
    if type(val) == type([]) and len(val) > 0:
        # print(key)
        val = val[:start_epoch]
        trlog[key] = val
        print(key, len(val))

# for key, val in trlog.items():
#     if type(val) == type([]) and len(val) > 0:
#         print(key, len(val))

    # print(type(val)) 
    # print(key)
# if __name__ == '__main__':
    # dataset = "WURS46"
    # method = "protonet"
    # n_shot = 5
    # trlog_path = "checkpoints/WURS46/ResNet12_s2m2_cs_aug_Adam_lr0.001_const_wd0_5way_5shot/trlog/20210423-033657"
    # # python save_features.py --method=cs_protonet --dataset=WURS46 --n_shot=5
    # # python test_s1.py --method=cs_protonet --dataset=WURS46 --n_shot=5

    # save_command = "python save_features.py --method=%s --dataset=%s --n_shot=%d" % (method, dataset, n_shot)
    # test_command = "python test_s1.py --method=%s --dataset=%s --n_shot=%d" % (method, dataset, n_shot)
    # fig_command = "python testss.py --file=%s" % (trlog_path)
    # # print(save_command) 
    # # print(test_command)
    # # os.system("ls")
    # os.system(fig_command)
    # # os.system(save_command)
    # # os.system(test_command)