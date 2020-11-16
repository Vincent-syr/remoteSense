import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from backbone import init_layer
from utils import euclidean_dist


class ProtoNetRotation(nn.Module):
    def __init__(self, model_func,  n_way, n_support, n_query, params):
        super(ProtoNetRotation, self).__init__()

        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = n_query
        self.feature = model_func()
        self.feat_dim   = self.feature.final_feat_dim
        self.loss_fn = nn.CrossEntropyLoss()