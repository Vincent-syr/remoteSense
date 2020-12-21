import torch
import matplotlib.pyplot as plt
from io_utils import save_fig
import numpy as np
# from io_utils import model_dict, parse_args, get_resume_file, get_trlog


# https://blog.csdn.net/TH_NUM/article/details/86105609



trlog_path = 'checkpoints/AIDS/ResNet10_rotate_aug_Adam_lr0.001_const_wd0_5way_5shot/trlog/20201119-231500'
save_fig(trlog_path)


# trlog_path = '/test/5Mycode/AttrMissing2/checkpoints/CUB/ResNet12_am3_aug_lr0.01_pwc_5way_5shot/trlog/20200927-163706'
# trlog = torch.load(trlog_path)
# attr_ratio = trlog['attr_ratio']
# img_ratio = 1 - np.array(attr_ratio)
# trlog['img_ratio'] = img_ratio
# print(img_ratio)
# # trlog.pop('attr_ratio')
# torch.save(trlog, trlog_path )
# save_fig(trlog_path)