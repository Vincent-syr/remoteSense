# totally am3, nothing to do with meta_template,
# used for multi_gpu traning



import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from backbone import init_layer
from utils import euclidean_dist



class ProtoNetMulti(nn.Module):
    def __init__(self, model_func,  n_way, n_support, n_query, params):
        super(ProtoNetMulti, self).__init__()

        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = n_query
        self.feature = model_func()
        self.feat_dim   = self.feature.final_feat_dim
        self.loss_fn = nn.CrossEntropyLoss()
        self.params = params
            


    def parse_feature(self, x, is_feature=False):
        if is_feature:
            z_all = x
        else:
            # x = x.view(self.n_way*(self.n_support+self.n_query), *x.size()[2:])
            # x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 

            z_all       = self.feature.forward(x)
            z_all       = z_all.view(self.n_way, self.n_support + self.n_query, -1)
            
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]

        return z_support, z_query


    # def correct(self, x, is_feature=False):
    def correct(self, scores):
        # scores.shape: (n_way * 4 query,  n_way)
        if self.params.method == 'rotate' and self.training:
            y_query = np.repeat(range(self.n_way ), 4 * self.n_query)
        else:
            y_query = np.repeat(range(self.n_way ), self.n_query)

        y_query = torch.from_numpy(y_query)
        y_query = Variable(y_query.long().cuda())

        pred = torch.argmax(scores, dim=1)
        correct = ((pred==y_query).sum()).item()
        count = scores.shape[0]
        return correct, count


        # if self.params.method == 'rotate' and self.training:
        #     y_query = np.repeat(range(self.n_way ), 4 * self.n_query)
        # else:
        #     y_query = np.repeat(range(self.n_way ), self.n_query)

        # topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        # topk_ind = topk_labels.cpu().numpy()
        # top1_correct = np.sum(topk_ind[:,0] == y_query)
        # return float(top1_correct), len(y_query)


    # def correct(self, z_all, lambda_c, attr_proj):
        
    #     y_query = np.repeat(range( self.n_way ), self.n_query )


    def correct_quick(self, z_all):
        """[summary] 

        Args:
            x ([type]): z_all is img feat extracted from backbone  (n_way, k+1, final_feat_dim)

        Returns:
            [type]: correct_this, count_this
        """

        scores = self.compute_score(z_all)
        correct_this, count_this = self.correct(scores)
        return correct_this, count_this


    # def forward(self, img_feat, attr_feat, is_feature=False):
    def forward(self, x):
        """[summary]

        Args:
            x[0] ([type]): image:  (n_way*(k_shot+query), 3, 224, 224)
            x[1] ([type]): attribute: (n_way, feat_dim=312)
        """

        z_all       = self.feature.forward(x)

        return z_all




    def compute_score(self, z_all):
        # z_all       = z_all.view(self.n_way, (self.n_support + self.n_query), -1)
        if self.params.method == 'rotate' and self.training:
            z_all       = z_all.view(self.n_way,  4*(self.n_support + self.n_query), -1)
            z_support   = z_all[:, :4 * self.n_support]
            z_query     = z_all[:, 4 * self.n_support:]
            # print("use rotation data")
        else:
            z_all       = z_all.view(self.n_way, (self.n_support + self.n_query), -1)
            z_support   = z_all[:, :self.n_support]
            z_query     = z_all[:, self.n_support:]

        z_query     = z_query.contiguous().view(-1, z_query.shape[-1])   # (n_way*query, feat_dim)
        img_proto   = z_support.mean(1)   # (n_way, feat_dim)
        dists = euclidean_dist(z_query, img_proto)
        scores = -dists
        return scores

        







