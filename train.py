import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
import copy
from methods import backbone, resnet12
from data.datamgr import SetDataManager
# from methods.protonet import ProtoNet
from methods.protonet_multi_gpu import ProtoNetMulti
from loss.nt_xent import NTXentLoss
from io_utils import model_dict, parse_args, get_resume_file, get_trlog, save_fig
from utils import Timer
from utils import euclidean_dist

import warnings
warnings.filterwarnings('ignore')


# torch.cuda.set_device(0)


def adjust_learning_rate(params, optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    if params.lr_anneal == 'const':
        pass
    elif params.lr_anneal == 'pwc':
        for param_group in optimizer.param_groups:
            if epoch >=300:
                param_group['lr'] = init_lr*0.1
            # elif epoch>=250:
            #     param_group['lr'] = init_lr*0.01
            # elif epoch ==300:
            #     param_group['lr'] = init_lr*0.001
    
    elif params.lr_anneal == 'exp':
        pass
    else:
        raise ValueError('No such lr anneal')


def compute_acc(scores, y):
    pred = torch.argmax(scores, dim=1)
    correct = ((pred==y).sum()).item()
    count = scores.shape[0]
    return correct, count


def train(base_loader, val_loader, model, start_epoch, stop_epoch, params, trlog):
    trlog_path = params.trlog_path

    init_lr = params.init_lr
    if params.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.001)
    elif params.optim == 'Adam':
        # optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
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
        iter_num = len(base_loader)

        for i, (x, _) in enumerate(base_loader, 1):
            x = x.cuda()   #  shape(n_way*(n_shot+query), 3, 224,224)
            z = model.forward(x)
            scores = model.compute_score(z)
            correct_this, count_this = model.correct(scores)

            y_query = torch.from_numpy(np.repeat(range(model.n_way), model.n_query))
            y_query = Variable(y_query.long().cuda())
            loss = model.loss_fn(scores, y_query)
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

        print('Train Acc = %4.2f%% +- %4.2f%%' %(acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        end = time.time()
        # print("train time = %.2f s" % (end-start))
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
                z = model.forward(x)
                scores = model.compute_score(z)
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
        print('epoch: ', epoch, 'lr: ', optimizer.param_groups[0]['lr'])


        if acc_mean > trlog['max_acc'] : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            trlog['max_acc_epoch'] = epoch
            trlog['max_acc']  = acc_mean

            print("best model! save...")
            outfile = os.path.join(params.model_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'max_acc':trlog['max_acc']}, outfile)

        # save model and trlog regularly
        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.model_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'max_acc': trlog['max_acc']}, outfile)
            torch.save(trlog, trlog_path)

            
        torch.cuda.empty_cache()  
        # best epoch and val acc
        print('best epoch = %d, best val acc = %.2f%%' % (int(trlog['max_acc_epoch']), trlog['max_acc']))
        # cumulative cost time / total time predicted
        print('ETA:{}/{}'.format(timer.measure(), timer.measure((epoch+1-start_epoch) / float(stop_epoch-start_epoch))))
        





def train_rotation(base_loader, val_loader,  model, start_epoch, stop_epoch, params, max_acc):
    
    trlog = get_trlog(params)
    trlog['max_acc'] = max_acc


    trlog_path = params.trlog_path
    rotate_classifier = nn.Sequential(nn.Linear(model.feat_dim, 4)) 
    rotate_classifier.cuda()

    init_lr = params.init_lr
    if params.optim == 'SGD':
        # optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.001)
       optimizer = torch.optim.SGD([
                {'params': model.parameters()},
                {'params': rotate_classifier.parameters()}], lr=init_lr, momentum=0.9, weight_decay=params.wd)

    elif params.optim == 'Adam':
        optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': rotate_classifier.parameters()}
        ], lr=init_lr)
            
    else:
        raise ValueError('Unknown Optimizer !!')

    timer = Timer()
    print_freq = 50
    lossfn = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, stop_epoch):
        adjust_learning_rate(params, optimizer, epoch, init_lr)
        model.train()
        start = time.time()
        cum_closs = 0
        cum_rloss = 0
        cacc_all = []
        racc_all = []
        iter_num = len(base_loader)

        for i, (x, _) in enumerate(base_loader, 1):
            # shape(n_way*(n_shot+query), 3, 224,224)
            bs = x.size(0)   # n_way * (k_shot+query)
            x_ = []
            a_ = []
            for j in range(bs):
                x90 = x[j].transpose(2,1).flip(1)
                x180 = x90.transpose(2,1).flip(1)
                x270 =  x180.transpose(2,1).flip(1)
                x_ += [x[j], x90, x180, x270]
                a_ += [torch.tensor(0),torch.tensor(1),torch.tensor(2),torch.tensor(3)]

            x_ = Variable(torch.stack(x_,0)).cuda()
            # y_ = Variable(torch.stack(y_,0))
            a_ = Variable(torch.stack(a_,0)).cuda()
            # forward
            z = model.forward(x_)
            scores = model.compute_score(z)
            correct_this, count_this = model.correct(scores)
            
            rotate_scores =  rotate_classifier(z)   
            rcorrect_this, rcount_this = compute_acc(rotate_scores, a_)

            y_query = torch.from_numpy(np.repeat(range(model.n_way), 4*model.n_query))
            y_query = Variable(y_query.long().cuda())
                
            rloss = lossfn(rotate_scores, a_)
            closs = model.loss_fn(scores, y_query)

            loss = 0.5*closs + 0.5*rloss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            cum_closs += closs.item()
            cum_rloss += rloss.item()
            avg_closs = cum_closs / float(i)
            avg_rloss = cum_rloss / float(i)

            # classfify accuracy
            cacc_all.append(correct_this/ float(count_this)*100)
            cacc_mean = np.array(cacc_all).mean()
            cacc_std = np.std(cacc_all)

            # rotate accuracy
            racc_all.append(rcorrect_this / float(rcount_this)*100)
            racc_mean = np.array(racc_all).mean()
            racc_std = np.std(racc_all)

            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Class Loss {:.3f} | Rotate Loss {:.3f}'.format(epoch, i, len(base_loader), avg_closs, avg_rloss))
                print('                     Class Acc  {:.3f} | Rotate Acc  {:.3f}'.format(cacc_mean, racc_mean))

        print('Train Acc = %4.2f%% +- %4.2f%%' %(cacc_mean, 1.96* cacc_std/np.sqrt(iter_num)))
        trlog['train_loss'].append(avg_closs)
        trlog['train_acc'].append(cacc_mean)

        trlog['train_rloss'].append(avg_rloss)
        trlog['train_racc'].append(racc_mean)
        trlog['epoch'].append(epoch)

        # evaluate
        model.eval()
        rotate_classifier.eval()
        iter_num = len(val_loader)
        with torch.no_grad():
            cacc_all = []
            racc_all = []
            for i, (x, _) in enumerate(val_loader, 1):
                # shape(n_way*(n_shot+query), 3, 224,224)
                x = x.cuda()
                z = model.forward(x)
                scores = model.compute_score(z)
                correct_this, count_this = model.correct(scores)

                cacc_all.append(correct_this/ float(count_this)*100)

            # classfify accuracy
            cacc_all  = np.array(cacc_all)
            cacc_mean = np.mean(cacc_all)
            cacc_std  = np.std(cacc_all)

            print('%d Test Acc = %4.2f%% +- %4.2f%%' %(
                iter_num,  cacc_mean, 1.96* cacc_std/np.sqrt(iter_num)))

            # print('%d Test Acc = %4.2f%% +- %4.2f%%, Rotate Acc = %4.2f' %(
            #     iter_num,  cacc_mean, 1.96* cacc_std/np.sqrt(iter_num), racc_mean))

            trlog['val_acc'].append(cacc_mean)
            # trlog['val_racc'].append(racc_mean)

        trlog['lr'].append(optimizer.param_groups[0]['lr'])
        print('epoch: ', epoch, 'lr: ', optimizer.param_groups[0]['lr'])

        if cacc_mean > trlog['max_acc'] : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            trlog['max_acc_epoch'] = epoch
            trlog['max_acc']  = cacc_mean

            print("best model! save...")
            outfile = os.path.join(params.model_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'max_acc':trlog['max_acc']}, outfile)

        # save model and trlog regularly
        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.model_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'max_acc': trlog['max_acc']}, outfile)
            torch.save(trlog, trlog_path)

        torch.cuda.empty_cache()  
        # best epoch and val acc
        print('best epoch = %d, best val acc = %.2f%%' % (int(trlog['max_acc_epoch']), trlog['max_acc']))
        # cumulative cost time / total time predicted
        print('ETA:{}/{}'.format(timer.measure(), timer.measure((epoch+1-start_epoch) / float(stop_epoch-start_epoch))))




def train_aug(base_loader, val_loader,  model, start_epoch, stop_epoch, params, max_acc):
    trlog = get_trlog(params)
    trlog['max_acc'] = max_acc

    trlog_path = params.trlog_path

    rotate_classifier = nn.Sequential(nn.Linear(model.module.feat_dim, 4)) 
    init_lr = params.init_lr
    if params.optim == 'SGD':
        # optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.001)
       optimizer = torch.optim.SGD([
                {'params': model.parameters()},
                {'params': rotate_classifier.parameters()}], lr=init_lr, momentum=0.9, weight_decay=params.wd)

    elif params.optim == 'Adam':
        optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': rotate_classifier.parameters()}
        ], lr=init_lr)
            
    else:
        raise ValueError('Unknown Optimizer !!')

    timer = Timer()
    print_freq = 50
    lossfn = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, stop_epoch):
        adjust_learning_rate(params, optimizer, epoch, init_lr)
        model.train()
        start = time.time()
        cum_closs = 0
        cum_rloss = 0
        cacc_all = []
        racc_all = []
        iter_num = len(base_loader)
        for i, (x, _) in enumerate(base_loader, 1):
            # shape(n_way*(n_shot+query), 3, 224,224)
            bs = x.size(0)   # n_way * (k_shot+query)
            x_ = []
            a_ = []
            for j in range(bs):
                x90 = x[j].transpose(2,1).flip(1)
                x180 = x90.transpose(2,1).flip(1)
                x270 =  x180.transpose(2,1).flip(1)
                x_ += [x[j], x90, x180, x270]
                # a_ += [torch.tensor(0),torch.tensor(1),torch.tensor(2),torch.tensor(3)]

            x_ = Variable(torch.stack(x_,0)).cuda()  # shape(4 * n_way * (n_shot+query), 3, 224,224)
            z = model.forward(x_)
            scores = model.compute_score(z)
            correct_this, count_this = model.correct(scores)
            








def train_totate_multiGPU(base_loader, val_loader,  model, start_epoch, stop_epoch, params, max_acc):
    trlog = get_trlog(params)
    trlog['max_acc'] = max_acc
    trlog_path = params.trlog_path

    rotate_classifier = nn.Sequential(nn.Linear(model.module.feat_dim, 4)) 
    rotate_classifier.cuda()

    init_lr = params.init_lr
    if params.optim == 'SGD':
        # optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.001)
       optimizer = torch.optim.SGD([
                {'params': model.parameters()},
                {'params': rotate_classifier.parameters()}], lr=init_lr, momentum=0.9, weight_decay=params.wd)

    elif params.optim == 'Adam':
        optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': rotate_classifier.parameters()}
        ], lr=init_lr)
            
    else:
        raise ValueError('Unknown Optimizer !!')

    timer = Timer()
    print_freq = 50
    lossfn = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, stop_epoch):
        adjust_learning_rate(params, optimizer, epoch, init_lr)
        model.train()
        start = time.time()
        cum_closs = 0
        cum_rloss = 0
        cacc_all = []
        racc_all = []
        iter_num = len(base_loader)

        for i, (x, _) in enumerate(base_loader, 1):
            # shape(n_way*(n_shot+query), 3, 224,224)
            bs = x.size(0)   # n_way * (k_shot+query)
            x_ = []
            a_ = []
            for j in range(bs):
                x90 = x[j].transpose(2,1).flip(1)
                x180 = x90.transpose(2,1).flip(1)
                x270 =  x180.transpose(2,1).flip(1)
                x_ += [x[j], x90, x180, x270]
                a_ += [torch.tensor(0),torch.tensor(1),torch.tensor(2),torch.tensor(3)]

            x_ = Variable(torch.stack(x_,0)).cuda()
            # y_ = Variable(torch.stack(y_,0))
            a_ = Variable(torch.stack(a_,0)).cuda()
            # forward
            z = model.forward(x_)
            scores = model.module.compute_score(z)
            correct_this, count_this = model.module.correct(scores)
            
            rotate_scores =  rotate_classifier(z)   
            rcorrect_this, rcount_this = compute_acc(rotate_scores, a_)

            y_query = torch.from_numpy(np.repeat(range(model.module.n_way), 4*model.module.n_query))
            y_query = Variable(y_query.long().cuda())

            rloss = lossfn(rotate_scores, a_)
            closs = model.module.loss_fn(scores, y_query)

            loss = 0.5*closs + 0.5*rloss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            cum_closs += closs.item()
            cum_rloss += rloss.item()
            avg_closs = cum_closs / float(i)
            avg_rloss = cum_rloss / float(i)

            # classfify accuracy
            cacc_all.append(correct_this/ float(count_this)*100)
            cacc_mean = np.array(cacc_all).mean()
            cacc_std = np.std(cacc_all)

            # rotate accuracy
            racc_all.append(rcorrect_this / float(rcount_this)*100)
            racc_mean = np.array(racc_all).mean()
            racc_std = np.std(racc_all)

            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Class Loss {:.3f} | Rotate Loss {:.3f}'.format(epoch, i, len(base_loader), avg_closs, avg_rloss))
                print('                     Class Acc  {:.3f} | Rotate Acc  {:.3f}'.format(cacc_mean, racc_mean))

        print('Train Acc = %4.2f%% +- %4.2f%%' %(cacc_mean, 1.96* cacc_std/np.sqrt(iter_num)))
        trlog['train_loss'].append(avg_closs)
        trlog['train_acc'].append(cacc_mean)

        trlog['train_rloss'].append(avg_rloss)
        trlog['train_racc'].append(racc_mean)
        trlog['epoch'].append(epoch)

        # evaluate
        model.eval()
        rotate_classifier.eval()
        iter_num = len(val_loader)
        with torch.no_grad():
            cacc_all = []
            racc_all = []
            for i, (x, _) in enumerate(base_loader, 1):
                # shape(n_way*(n_shot+query), 3, 224,224)
                # bs = x.size(0)   # n_way * (k_shot+query)
                # x_ = []
                # a_ = []
                # for j in range(bs):
                #     x90 = x[j].transpose(2,1).flip(1)
                #     x180 = x90.transpose(2,1).flip(1)
                #     x270 =  x180.transpose(2,1).flip(1)
                #     x_ += [x[j], x90, x180, x270]
                #     a_ += [torch.tensor(0),torch.tensor(1),torch.tensor(2),torch.tensor(3)]

                # x_ = Variable(torch.stack(x_,0)).cuda()
                # a_ = Variable(torch.stack(a_,0)).cuda()

                x = x.cuda()
                z = model.forward(x)
                scores = model.module.compute_score(z)
                correct_this, count_this = model.module.correct(scores)

                # rotate_scores =  rotate_classifier(z)   
                # rcorrect_this, rcount_this = compute_acc(rotate_scores, a_)

                cacc_all.append(correct_this/ float(count_this)*100)
                # racc_all.append(rcorrect_this/ float(rcount_this)*100)

            # classfify accuracy
            cacc_all  = np.array(cacc_all)
            cacc_mean = np.mean(cacc_all)
            cacc_std  = np.std(cacc_all)

            # # rotate accuracy
            # racc_all.append(rcorrect_this / float(rcount_this)*100)
            # racc_mean = np.array(racc_all).mean()
            # racc_std = np.std(racc_all)

            print('%d Test Acc = %4.2f%% +- %4.2f%%' %(
                iter_num,  cacc_mean, 1.96* cacc_std/np.sqrt(iter_num)))

            # print('%d Test Acc = %4.2f%% +- %4.2f%%, Rotate Acc = %4.2f' %(
            #     iter_num,  cacc_mean, 1.96* cacc_std/np.sqrt(iter_num), racc_mean))

            trlog['val_acc'].append(cacc_mean)
            # trlog['val_racc'].append(racc_mean)

        trlog['lr'].append(optimizer.param_groups[0]['lr'])
        print('epoch: ', epoch, 'lr: ', optimizer.param_groups[0]['lr'])

        if cacc_mean > trlog['max_acc'] : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            trlog['max_acc_epoch'] = epoch
            trlog['max_acc']  = cacc_mean

            print("best model! save...")
            outfile = os.path.join(params.model_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'max_acc':trlog['max_acc']}, outfile)

        # save model and trlog regularly
        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.model_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'max_acc': trlog['max_acc']}, outfile)
            torch.save(trlog, trlog_path)

        torch.cuda.empty_cache()  
        # best epoch and val acc
        print('best epoch = %d, best val acc = %.2f%%' % (int(trlog['max_acc_epoch']), trlog['max_acc']))
        # cumulative cost time / total time predicted
        print('ETA:{}/{}'.format(timer.measure(), timer.measure((epoch+1-start_epoch) / float(stop_epoch-start_epoch))))


def train_contrastive(base_loader, val_loader,  model, start_epoch, stop_epoch, params, trlog):
    trlog_path = params.trlog_path

    cs_mlp = nn.Sequential(
        nn.Linear(model.feat_dim, model.feat_dim),
        nn.ReLU(), 
        nn.Linear(model.feat_dim, 256)
    ).cuda()

    cs_criterion = NTXentLoss(params.batch_size).cuda()
    
    init_lr = params.init_lr

    if params.optim == 'SGD':
        optimizer = torch.optim.SGD([
                {'params': model.parameters()},
                {'params': cs_mlp.parameters()}
            ], lr=init_lr, momentum=0.9, weight_decay=params.wd)

    elif params.optim == 'Adam':
        optimizer = torch.optim.Adam([
                    {'params': model.parameters()},
                    {'params': cs_mlp.parameters()}
            ], lr=init_lr)
    else:
        raise ValueError('Unknown Optimizer !!')

    timer = Timer()
    print_freq = 50

    for epoch in range(start_epoch, stop_epoch):

        adjust_learning_rate(params, optimizer, epoch, init_lr)
        model.train()
        start = time.time()
        cum_clf_loss = 0
        cum_cs_loss = 0
        cacc_all = []
        iter_num = len(base_loader)
        for i, (x, _) in enumerate(base_loader, 1):
            xi = x[0].cuda()    # n_way * (k_shot+query)
            xj = x[1].cuda()    # n_way * (k_shot+query)
            # compute clf loss and acc
            zi = model.forward(xi)   # 只算xi的loss
            zj = model.forward(xj)  
            scores = model.compute_score(zi)
            correct_this, count_this = model.correct(scores)
            y_query = torch.from_numpy(np.repeat(range(model.n_way), model.n_query))
            y_query = Variable(y_query.long().cuda())
            clf_loss = model.loss_fn(scores, y_query)
            
            # compute contrastive loss
            zics = cs_mlp(zi)
            zjcs = cs_mlp(zj)

            cs_loss = cs_criterion(zics, zjcs)
            loss = clf_loss + cs_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            cum_clf_loss += clf_loss.item()
            cum_cs_loss += cs_loss.item()
            avg_clf_loss = cum_clf_loss / i
            avg_cs_loss = cum_cs_loss / i
            # classfify accuracy
            cacc_all.append(correct_this/ float(count_this)*100)
            cacc_mean = np.array(cacc_all).mean()
            cacc_std = np.std(cacc_all)

            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Clf_lass Loss {:.3f} | Contrastive Loss {:.3f}'.format(epoch, i, len(base_loader), avg_clf_loss, avg_cs_loss))
                print('                     Class Acc  {:.3f}'.format(cacc_mean))

        print('Train Acc = %4.2f%% +- %4.2f%%' %(cacc_mean, 1.96* cacc_std/np.sqrt(iter_num)))
        trlog['train_loss'].append(avg_clf_loss)
        trlog['train_cs_loss'].append(avg_cs_loss)
        trlog['train_acc'].append(cacc_mean)

        # val
        model.eval()
        start = time.time()
        iter_num = len(val_loader)
        with torch.no_grad():
            acc_all = []
            for i, (x, _) in enumerate(val_loader, 1):
                x = x.cuda()    # n_way * (k_shot+query)
                z = model.forward(x)
                scores = model.compute_score(z)
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
        print('epoch: ', epoch, 'lr: ', optimizer.param_groups[0]['lr'])

        if acc_mean > trlog['max_acc'] : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            trlog['max_acc_epoch'] = epoch
            trlog['max_acc']  = acc_mean

            print("best model! save...")
            outfile = os.path.join(params.model_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'max_acc':trlog['max_acc']}, outfile)

        # save model and trlog regularly
        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.model_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'max_acc': trlog['max_acc']}, outfile)
            torch.save(trlog, trlog_path)
    
        torch.cuda.empty_cache()  
        # best epoch and val acc
        print('best epoch = %d, best val acc = %.2f%%' % (int(trlog['max_acc_epoch']), trlog['max_acc']))
        # cumulative cost time / total time predicted
        print('ETA:{}/{}'.format(timer.measure(), timer.measure((epoch+1-start_epoch) / float(stop_epoch-start_epoch))))




def train_s2m2_cs(base_loader, val_loader,  model, start_epoch, stop_epoch, params, trlog):

    trlog_path = params.trlog_path

    cs_mlp = nn.Sequential(
        nn.Linear(model.feat_dim, model.feat_dim),
        nn.ReLU(), 
        nn.Linear(model.feat_dim, 256)
    ).cuda()

    cs_criterion = NTXentLoss(params.batch_size).cuda()
    
    init_lr = params.init_lr

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    if params.optim == 'SGD':
        optimizer = torch.optim.SGD([
                {'params': model.parameters()},
                {'params': cs_mlp.parameters()}
            ], lr=init_lr, momentum=0.9, weight_decay=params.wd)

    elif params.optim == 'Adam':
        optimizer = torch.optim.Adam([
                    {'params': model.parameters()},
                    {'params': cs_mlp.parameters()}
            ], lr=init_lr)
    else:
        raise ValueError('Unknown Optimizer !!')

    timer = Timer()
    print_freq = 50

    for epoch in range(start_epoch, stop_epoch):

        adjust_learning_rate(params, optimizer, epoch, init_lr)
        model.train()
        start = time.time()
        cum_clf_loss = 0
        cum_cs_loss = 0
        cum_mm_loss = 0
        cacc_all = []
        iter_num = len(base_loader)

        for i, (x, _) in enumerate(base_loader, 1):
            # manifold mixup loss
            xi = x[0].cuda()    # n_way * (k_shot+query)
            xj = x[1].cuda()    # n_way * (k_shot+query)
            # manifold mixup loss
            inputs = xi
            targets = torch.from_numpy(np.repeat(range(model.n_way), model.n_query)).cuda()
            lam = np.random.beta(params.alpha, params.alpha)
            # 分割 x_support, x_query
            inputs = inputs.view(model.n_way, (model.n_support + model.n_query), 3, params.image_size, params.image_size)

            x_support   = inputs[:, :model.n_support]   # (n_way, n_support, )
            x_query     = inputs[:, model.n_support:]  
            x_support   = x_support.contiguous().view(model.n_way*model.n_support, 3, params.image_size, params.image_size)  # (n_way * n_support) 
            x_query     = x_query.contiguous().view(model.n_way*model.n_query, 3, params.image_size, params.image_size)  # (n_way * n_query) 
            # x_support：model forward：得到z_support
            z_support = model.forward(x_support)
            z_support = z_support.view(model.n_way, model.n_support, -1)
            # x_query 计算插值, 得到z_query, target_a, target_b
            z_query, target_a , target_b = model.forward(x_query, targets, mixup_hidden= True, lam = lam)  # (n_way * n_query, feat_dim)
            img_proto   = z_support.mean(1)   # (n_way, feat_dim)
            # 用z_pred，tareget_a, target_b计算loss
            dists = euclidean_dist(z_query, img_proto)  # (n_way*n_query, n_way)
            scores = -dists
            criterion = model.loss_fn
            mm_loss = mixup_criterion(criterion, scores, target_a, target_b, lam)
            
            mm_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cum_mm_loss = mm_loss.item()
            avg_mm_loss = cum_mm_loss / float(i)
            # train_loss += loss.data.item()

            # compute clf_loss and cs_loss
            zi = model.forward(xi)   # 只算xi的loss
            zj = model.forward(xj)  
            scores = model.compute_score(zi)
            correct_this, count_this = model.correct(scores)
            y_query = torch.from_numpy(np.repeat(range(model.n_way), model.n_query))
            y_query = Variable(y_query.long().cuda())
            clf_loss = model.loss_fn(scores, y_query)

            # compute contrastive loss
            zics = cs_mlp(zi)
            zjcs = cs_mlp(zj)

            cs_loss = cs_criterion(zics, zjcs)
            loss = clf_loss + cs_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            cum_clf_loss += clf_loss.item()
            cum_cs_loss += cs_loss.item()
            avg_clf_loss = cum_clf_loss / i
            avg_cs_loss = cum_cs_loss / i
            # classfify accuracy
            cacc_all.append(correct_this/ float(count_this)*100)
            cacc_mean = np.array(cacc_all).mean()
            cacc_std = np.std(cacc_all)

            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Clf_lass Loss {:.3f} | Contrastive Loss {:.3f}'.format(epoch, i, len(base_loader), avg_clf_loss, avg_cs_loss))
                print('Mixup Loss  {:.3f} |   Class Acc  {:.3f}'.format(avg_mm_loss, cacc_mean))

        print('Train Acc = %4.2f%% +- %4.2f%%' %(cacc_mean, 1.96* cacc_std/np.sqrt(iter_num)))
        trlog['train_loss'].append(avg_clf_loss)
        trlog['train_cs_loss'].append(avg_cs_loss)
        trlog['train_mm_loss'].append(avg_mm_loss)
        trlog['train_acc'].append(cacc_mean)


        # val
        model.eval()
        start = time.time()
        iter_num = len(val_loader)
        with torch.no_grad():
            acc_all = []
            for i, (x, _) in enumerate(val_loader, 1):
                x = x.cuda()    # n_way * (k_shot+query)
                z = model.forward(x)
                scores = model.compute_score(z)
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
        print('epoch: ', epoch, 'lr: ', optimizer.param_groups[0]['lr'])

        if acc_mean > trlog['max_acc'] : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            trlog['max_acc_epoch'] = epoch
            trlog['max_acc']  = acc_mean

            print("best model! save...")
            outfile = os.path.join(params.model_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'max_acc':trlog['max_acc']}, outfile)

        # save model and trlog regularly
        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.model_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'max_acc': trlog['max_acc']}, outfile)
            torch.save(trlog, trlog_path)

        torch.cuda.empty_cache()  
        # best epoch and val acc
        print('best epoch = %d, best val acc = %.2f%%' % (int(trlog['max_acc_epoch']), trlog['max_acc']))
        # cumulative cost time / total time predicted
        print('ETA:{}/{}'.format(timer.measure(), timer.measure((epoch+1-start_epoch) / float(stop_epoch-start_epoch))))


def train_s2m2_sc(base_loader, val_loader,  model, start_epoch, stop_epoch, params, trlog):
    # protonet + supervised contrastive loss + mixup loss

    pass

def train_baseline(base_loader, val_loader,  model, start_epoch, stop_epoch, params, trlog):
    
    
    pass




if __name__ == "__main__":
    np.random.seed(10)
    params = parse_args('train')
    print(params)    

    # if params.os == 'linux':
    #     base_file = os.path.join('./filelists', params.dataset, 'base_linux.json')
    #     val_file = os.path.join('./filelists', params.dataset, 'val_linux.json')
    # else:
    base_file = os.path.join('./filelists', params.dataset, ('base_%s.json' % params.os))
    val_file = os.path.join('./filelists', params.dataset,  ('val_%s.json' % params.os))
    # novel_file = os.path.join('./filelists', params.dataset, 'novel.json')

    image_size = 84
    params.image_size = image_size
    print('image_size = ', image_size)
    print("n_query = ", params.n_query)
    params.batch_size = (params.n_query + params.n_shot) * params.train_n_way

    train_few_shot_params   = dict(n_way = params.train_n_way, n_support = params.n_shot, n_query=params.n_query) 
    base_datamgr            = SetDataManager(image_size, n_episode=params.n_episode, params=params, **train_few_shot_params)
    base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )

    test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot, n_query = params.n_query) 
    val_datamgr             = SetDataManager(image_size, n_episode=params.n_episode, params=params, **test_few_shot_params)
    val_loader              = val_datamgr.get_data_loader(val_file, aug = False) 

    # novel_datamgr             = SetDataManager(image_size, n_episode=params.n_episode , **test_few_shot_params)
    # novel_loader              = val_datamgr.get_data_loader(novel_file, aug = False) 

    # model = ProtoNet(model_dict[params.model], params=params, **train_few_shot_params)
    if params.method == 'protonet' or params.method == 'rotate' or 'cs_protonet':
        model = ProtoNetMulti(model_dict[params.model], params=params, **train_few_shot_params)
    # elif params.method = 'rotate':
    else:
        raise ValueError('Unknown method')
    model = model.cuda()
    params.checkpoint_dir = 'checkpoints/%s/%s_%s' %(params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'

    params.checkpoint_dir += '_%s_lr%s_%s_wd%s' % (params.optim, str(params.init_lr), params.lr_anneal, str(params.wd))

    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
    params.model_dir = os.path.join(params.checkpoint_dir, 'model')


    if not os.path.isdir(params.model_dir):
        os.makedirs(params.model_dir)
    print('checkpoint_dir = ', params.checkpoint_dir)
    
    # print(params.train_aug)
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
            # trlog load
            old_trlog_path = os.path.join(params.checkpoint_dir, 'trlog', params.trlog_file)
            trlog = torch.load(old_trlog_path)      # 只截取当前0~start epoch
            trlog['args'] = vars(params)
            params.trlog_path = os.path.join(params.checkpoint_dir, 'trlog', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
            for key, val in trlog.items():
                if type(val) == type([]) and len(val) > 0:
                    print(key)
                    val = val[:start_epoch]
                    trlog[key] = val
            print('resume training from ', params.start_epoch)
        else:
            raise ValueError("no such resume file")
    
    # training from scratch
    else:
        trlog = get_trlog(params)
        trlog['max_acc'] = max_acc
        trlog_dir = os.path.join(params.checkpoint_dir, 'trlog')
        if not os.path.isdir(trlog_dir):
            os.makedirs(trlog_dir)
        params.trlog_path = os.path.join(trlog_dir, time.strftime("%Y%m%d-%H%M%S", time.localtime()))   # '20200909-185444'



    if params.method == 'rotate':
        train_rotation(base_loader, val_loader,  model, start_epoch, stop_epoch, params, trlog)
    elif params.method == 'protonet':
        train(base_loader, val_loader,  model, start_epoch, stop_epoch, params, trlog)
    elif params.method == "cs_protonet":
        train_contrastive(base_loader, val_loader,  model, start_epoch, stop_epoch, params, trlog)
    elif params.method == 's2m2_cs':
        train_s2m2_cs(base_loader, val_loader,  model, start_epoch, stop_epoch, params, trlog)
    elif params.method == "baseline":
        train_baseline(base_loader, val_loader,  model, start_epoch, stop_epoch, params, trlog)
    else:
        raise ValueError("no such method")
        
    # draw picture
    fig_command = "python testss.py --file=%s" % (params.trlog_path)
    os.system(fig_command)

    # after training, save feature and test
    save_command = "python save_features.py --method=%s --dataset=%s --n_shot=%d --lr_anneal=%s --optim=%s" % (
                                        params.method, params.dataset, params.n_shot, params.lr_anneal, params.optim)
    test_command = "python test_s1.py --method=%s --dataset=%s --n_shot=%d --lr_anneal=%s --optim=%s" % (
                                        params.method, params.dataset, params.n_shot, params.lr_anneal, params.optim)
    # print(save_command)   
    # print(test_command)   
    os.system(save_command)
    os.system(test_command)


