import numpy as np
import torch
from sklearn import preprocessing
import sys
import os
from pathlib import Path
import pickle
import copy

import data.feature_loader as feat_loader


class DATA_LOADER(object):
    def __init__(self, dataset, data_file, attr_file, n_way, k_shot):

        # self.dataset = dataset
        # self.auxiliary_data_source = aux_datasource   # attribute
        # self.all_data_sources = ['resnet_features'] + [self.auxiliary_data_source]
        print('data_file = ', data_file)
        print('attr_file =', attr_file)
        self.k_shot = k_shot
        self.n_way = n_way
        self.cl_data_file = feat_loader.init_loader(data_file)
        self.class_unique = list(self.cl_data_file.keys())
        if attr_file:
            self.aux = True
            if dataset == 'CUB':
                self.aux_data = torch.from_numpy(np.load(attr_file))
            elif dataset == 'miniImagenet':
                self.aux_data = torch.from_numpy(np.load(attr_file)['features'].astype('float32'))
            self.attr_dim = self.aux_data.shape[1]
        else:
            self.aux = False
        tmp = self.cl_data_file[self.class_unique[0]][0]
        self.img_dim = len(tmp)

    def next_batch(self, batch_size=1):
        #####################################################################
        # gets batch from train_feature = 7057 samples from 100 train classes
        #####################################################################
        idxes = torch.randperm(len(self.class_unique))[:self.n_way*batch_size]
        classes = [self.class_unique[i] for i in idxes]
        batch_feature = []
        for c in classes:
            l = torch.tensor(self.cl_data_file[c])
            pos = torch.randperm(len(l))[:self.k_shot]
            feat = l[pos]
            batch_feature.append(feat)

        # batch_feature = torch.tensor(batch)   # (n*b, k, f1)
        batch_feature = [t.cpu().numpy() for t in batch_feature]   # [tensor --> ndarray]
        batch_feature = torch.tensor(batch_feature)  # list --> tensor
        batch_label = classes              # (n*b)
        if self.aux:
            batch_att = self.aux_data[batch_label]   # (n*b, f2)
            batch = [batch_feature, batch_att]
        else:
            batch = batch_feature

        return batch_label, batch
