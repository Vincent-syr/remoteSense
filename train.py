import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

import time
import os
import glob
import copy
# import configs
import backbone
from data.datamgr import SetDataManager
import argparse

from io_utils import model_dict, parse_args, get_resume_file, get_trlog, save_fig
from utils import Timer
from methods.protonet import ProtoNet

import warnings
warnings.filterwarnings('ignore')

def adjust_learning_rate(params, optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    if params.lr_anneal == 'const':
        pass
    elif params.lr_anneal == 'pwc':
        for param_group in optimizer.param_groups:
            if epoch >=200 and epoch < 250:
                param_group['lr'] = init_lr*0.1
            elif epoch>=250:
                param_group['lr'] = init_lr*0.01
            # elif epoch ==300:
            #     param_group['lr'] = init_lr*0.001
    
    elif params.lr_anneal == 'exp':
        pass


def train(base_loader, val_loader, model, start_epoch, stop_epoch, params, max_acc=0):
    trlog = get_trlog(params)
    trlog['max_acc'] = max_acc
    trlog_dir = os.path.join(params.checkpoint_dir, 'trlog')
    if not os.path.isdir(trlog_dir):
        os.makedirs(trlog_dir)
    trlog_path = os.path.join(trlog_dir, time.strftime("%Y%m%d-%H%M%S", time.localtime()))   # '20200909-185444'



    init_lr = params.init_lr
    if params.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.001)
    elif params.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=0.001)
    else:
        raise ValueError('Unknown Optimizer !!')

    timer = Timer()
    print_freq = 50

    for epoch in range(start_epoch, stop_epoch):
        adjust_learning_rate(params, optimizer, epoch, init_lr)
        model.train()
        start = time.time()
        cum_loss = 0
        acc_all = []
        for i, (x, y) in enumerate(base_loader, 1):
            x = x.cuda()   # shape(n_way*(n_shot+query), 3, 224,224)
            y = y.cuda()
            # print('x.shape = ', x.shape)
            z = model.forward(x)
            scores = model.compute_score(z)
            correct_this, count_this = model.correct(scores)

            y_query = torch.from_numpy(np.repeat(range(model.n_way), model.n_query))
            y_query = Variable(y_query.long().cuda())
            loss = model.loss_cls(scores, y_query)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            cum_loss = cum_loss + loss.item()
            avg_loss = cum_loss/float(i)
            acc_all.append(correct_this/ float(count_this)*100)

            acc_mean = np.array(acc_all).mean()
            acc_std = np.std(acc_all)

            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Acc {:.2f}'.format(
                    epoch, i, len(base_loader), avg_loss, acc_mean))
                # print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Acc {:.2f}%'.format(epoch, i, len(base_loader), avg_loss, acc_mean))

        print('Train Acc_cls = %4.2f%% +- %4.2f%%' %(acc_mean, acc_std))
        end = time.time()
        print("train time = %.2f s" % (end-start))
        trlog['train_loss'].append(avg_loss)
        trlog['train_acc'].append(acc_mean)
        trlog['epoch'].append(epoch)

        model.eval()
        start = time.time()
        iter_num = len(val_loader)
        with torch.no_grad():
            acc_all = []
            for x,_ in val_loader:
                x = x.cuda()
                z_all = model.forward(x)
                scores = model.compute_score(z_all)
                correct_this, count_this = model.correct(scores)

                acc_all.append(correct_this/ float(count_this)*100)
            acc_all  = np.array(acc_all)
            acc_mean = np.mean(acc_all)
            acc_std  = np.std(acc_all)
            print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
            trlog['val_acc'].append(acc_mean)
        end = time.time()
        print("validation time = %.2f s" % (end-start))
        trlog['lr'].append(optimizer.param_groups[0]['lr'])
        print("learning rate = ", optimizer.param_groups[0]['lr'])

        if acc_mean > trlog['max_acc'] : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            trlog['max_acc_epoch'] = epoch
            trlog['max_acc']  = acc_mean

            print("best model! save...")
            outfile = os.path.join(params.model_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'max_acc':max_acc}, outfile)

        # save model and trlog regularly
        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.model_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'max_acc': max_acc}, outfile)
            torch.save(trlog, trlog_path)

        torch.cuda.empty_cache()  
        # best epoch and val acc
        print('best epoch = %d, best val acc = %.2f%%' % (int(trlog['max_acc_epoch']), trlog['max_acc']))
        # cumulative cost time / total time predicted
        print('ETA:{}/{}'.format(timer.measure(), timer.measure((epoch+1-start_epoch) / float(stop_epoch-start_epoch))))




if __name__ == "__main__":
    np.random.seed(10)
    params = parse_args('train')
    print(params)
    
    # base_file = configs.data_dir[params.dataset] + 'base.json' 
    # val_file   = configs.data_dir[params.dataset] + 'val.json' 

    base_file = os.path.join('./filelists', params.dataset, 'base.json')
    val_file = os.path.join('./filelists', params.dataset, 'val.json')

    # if params.dataset == 'CUB':
    image_size = 224
    print('image_size = ', image_size)
    print("n_query = ", params.n_query)

    train_few_shot_params   = dict(n_way = params.train_n_way, n_support = params.n_shot, n_query=params.n_query) 

    base_datamgr            = SetDataManager(image_size, n_episode=params.n_episode, **train_few_shot_params)
    base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )

    test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot, n_query = params.n_query) 
    val_datamgr             = SetDataManager(image_size, n_episode=params.n_episode , **test_few_shot_params)
    val_loader              = val_datamgr.get_data_loader(val_file, aug = False) 


    model = ProtoNet(model_dict[params.model], params=params, **train_few_shot_params)
    model = model.cuda()
    params.checkpoint_dir = 'checkpoints/%s/%s_%s' %(params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'

    params.checkpoint_dir += '_lr%s_%s' % (str(params.init_lr), params.lr_anneal)

    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
    params.model_dir = os.path.join(params.checkpoint_dir, 'model')

    if not os.path.isdir(params.model_dir):
        os.makedirs(params.model_dir)
    print('checkpoint_dir = ', params.checkpoint_dir)
    # exit(0)
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    max_acc = 0


    if params.start_epoch != 0:
        # resume_file = get_resume_file(params.checkpoint_dir)
        resume_file = os.path.join(params.model_dir, str(params.start_epoch) +'.tar')
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            if 'max_acc' in tmp:
                max_acc = tmp['max_acc']
            model.load_state_dict(tmp['state'])
            print('resume training from ', params.start_epoch)
        else:
            raise ValueError("no such resume file")

    train(base_loader, val_loader,  model, start_epoch, stop_epoch, params, max_acc)
