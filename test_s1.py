'''
    test novel classes, use pre-train setting.
    non attribute missing
'''


import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import random
import time

# import configs
from methods import backbone, resnet12
import data.feature_loader as feat_loader
from data.dataloader_vae import DATA_LOADER
from data.datamgr import SimpleDataManager, SetDataManager
from io_utils import model_dict, parse_args, get_resume_file, get_best_file , get_assigned_file, save_fig, get_trlog_test
# from methods.protonet import ProtoNet
# from methods.am3_protonet import AM3_ProtoNet
# from methods.am3 import AM3
from methods.protonet_multi_gpu import ProtoNetMulti
import warnings

warnings.filterwarnings('ignore')


def evaluation(dataloader, model, split, trlog, params):
    # acc_all = {'base':[], 'val':[], 'novel':[]}
    acc_all = []
    if params.source == 'feature':
        for i in range(params.iter_num):
            _, data_from_modalities = dataloader.next_batch()
            if len(data_from_modalities) == 2:  # use attribute
                img_feat = data_from_modalities[0].cuda()  # (n*b, k+q, f1)
                attr_feat = data_from_modalities[1].cuda()  # (n*b, f2)

                z_all = [ img_feat, attr_feat]

            else: # only image feature
                z_all = data_from_modalities.cuda()

            correct_this, count_this = model.correct_quick(z_all)
        
            acc = correct_this/float(count_this) * 100
            trlog['%s_acc' % split].append(acc)
            acc_all.append(acc)

                # acc = feature_evaluation(dataset, model, adaptation=params.adaptation, **few_shot_params)
                
                # trlog['%s_acc' % split].append(acc)

    elif params.source == 'image':
        for x,_ in dataloader:
            if params.aux:
                
                x[0] = x[0].cuda()   # shape(n_way*(n_shot+query), 3, 224,224)
                x[1] = x[1].view(model.n_way, -1, x[1].shape[-1]).cuda()
                x[1] = x[1].mean(1)   # (n_way, feat_dim)            
                z_all, lambda_c, attr_proj = model.forward(x)
                scores = model.compute_score(z_all, lambda_c, attr_proj)

            else:
                x = x.cuda()
                z_all = model.forward(x)
                scores = model.compute_score(z_all)

            correct_this, count_this = model.correct(scores)
            acc = correct_this/ float(count_this)*100
            trlog['%s_acc' % split].append(acc)
            acc_all.append(acc)

    else:
        raise ValueError("unknown data source")

    return acc_all




def get_dataloader(model, split, params):
    if params.source == 'feature':
        image_file = os.path.join(params.checkpoint_dir.replace("checkpoints", "features"),
                                split + "_best.hdf5")  # defaut split = novel, but you can also test base or val classes
        attr_file = None
        if params.aux:
            if params.dataset == 'CUB':
                attr_file = 'filelists/CUB/attr_array.npy'
            elif params.dataset == 'miniImagenet':
                attr_file = '/test/0Dataset_others/Dataset/mini-imagenet-am3/few-shot-wordemb-{:}.npz'.format(split)

        dataloader = DATA_LOADER(params.dataset, image_file, attr_file,  
                                params.test_n_way, params.n_shot + n_query)
    #  dataset, data_file, attr_file, n_way, k_shot,aux_datasource='attributes'


    elif params.source == 'image':
    #         base_datamgr            = SetDataManager(image_size, aux=aux, n_episode=params.n_episode, **train_few_shot_params)
    # base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        few_shot_params   = dict(n_way = params.train_n_way, n_support = params.n_shot, n_query=params.n_query) 
        data_file = configs.data_dir[params.dataset] + '%s.json' % (split) 

        if params.aux:
            attr_file = configs.data_dir[params.dataset] + 'attr_array.npy'
            data_file = [data_file, attr_file]
        datamgr      = SetDataManager(image_size, aux=params.aux, n_episode=params.n_episode, **few_shot_params)
        dataloader  = datamgr.get_data_loader(data_file , aug = False)
    
    else:
        raise ValueError("unknown data source")
    return dataloader



if __name__ == '__main__':
    params = parse_args('test')
    print(params)
    params.iter_num = 400
    
    params.n_episode = params.iter_num
    image_size = 224
    n_query = params.n_query
    
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot, n_query=params.n_query)   # 5 way, 5 shot
    if params.method == 'protonet' or params.method == 'rotate':
        params.aux = False
        model = ProtoNetMulti(model_dict[params.model], params=params, **few_shot_params)
    elif params.method == 'am3':
        params.aux = True
        model = AM3(model_dict[params.model], params=params, word_dim=word_dim,  **few_shot_params)
    else:
        raise ValueError('Unknown method')
    
    model = model.cuda()
    model = model.eval()

    params.checkpoint_dir = 'checkpoints/%s/%s_%s' %(params.dataset, params.model, params.method)

    
    if params.train_aug:
        params.checkpoint_dir += '_aug'

    params.checkpoint_dir += '_%s_lr%s_%s_wd%s' % (params.optim, str(params.init_lr), params.lr_anneal, str(params.wd))
    
    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
    params.model_dir = os.path.join(params.checkpoint_dir, 'model')
    # params.record_dir = params.checkpoint_dir.replace("checkpoints", "record")
    params.record_dir = os.path.join(params.checkpoint_dir, "trlog")

    if not os.path.isdir(params.record_dir):
        os.makedirs(params.record_dir)

    record_file = os.path.join(params.record_dir, 'results.txt')

    trlog = get_trlog_test(params)
    trlog_path = os.path.join(params.record_dir, time.strftime("%Y%m%d-%H%M%S", time.localtime()))

    modelfile   = get_best_file(params.model_dir)
    if modelfile is not None:
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])
        print('load model ')

    print("checkpoint_dir = ", params.checkpoint_dir)
    print("record_dir = ", params.record_dir)

    split_list = ['base', 'val', 'novel']


    for split in split_list:
        dataloader = get_dataloader(model, split, params)
        
        acc_all = evaluation(dataloader, model, split, trlog, params)

        acc_array = np.asarray(acc_all)
        acc_mean = np.mean(acc_array)
        acc_std = np.std(acc_all)
        # writer.close()
        # print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        print('%d %s Acc = %4.2f%% +- %4.2f%%' % (params.iter_num, split, acc_mean, 1.96 * acc_std / np.sqrt(params.iter_num)))

        with open(record_file , 'a') as f:
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            aug_str = '-aug' if params.train_aug else ''
            aug_str += '-adapted' if params.adaptation else ''
            if params.method in ['baseline', 'baseline++'] :
                exp_setting = '%s-%s-%s-%s-%s%s %sshot %sway_test' %(params.dataset, params.source, split, params.model, params.method, aug_str, params.n_shot, params.test_n_way )
            else:
                exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' %(params.dataset, split, params.model, params.method, aug_str , params.n_shot , params.train_n_way, params.test_n_way )
            acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' %(params.iter_num, acc_mean, 1.96* acc_std/np.sqrt(params.iter_num))
            f.write( 'Time: %s, Setting: %s, Acc: %s \n' %(timestamp,exp_setting,acc_str)  )

    # torch.save(trlog, trlog_path)
    

