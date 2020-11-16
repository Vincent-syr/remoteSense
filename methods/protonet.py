# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from utils import euclidean_dist

# from methods.meta_template import MetaTemplate

class ProtoNet(nn.Module):
    def __init__(self, model_func,  n_way, n_support, n_query, params):
        super(ProtoNet, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = n_query
        self.feature = model_func()
        # print(self.feature)
        self.feat_dim   = self.feature.final_feat_dim
        self.loss_fn = nn.CrossEntropyLoss()


    def forward(self, x):
        z = self.feature.forward(x)
        return z

    def compute_score(self, z_all):
        z_all       = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]  # (n_way, n_query, feat_dim)
        z_query     = z_query.contiguous().view(-1, z_query.shape[-1])
        
        img_proto   = z_support.mean(1)   # (n_way, feat_dim)
        dists = euclidean_dist(z_query, img_proto)
        scores = -dists
        return scores

    def correct(self, scores):  
        # scores: (n_way*n_query , n_way) 每个query 到每个 prototype的距离
        y_query = np.repeat(range(self.n_way ), self.n_query )  # (n_way*n_query)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)