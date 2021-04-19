import numpy as np
import torch
from torch.autograd import Variable
import os
import glob
import h5py

# import configs
from methods import backbone, resnet12
from data.datamgr import SimpleDataManager
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file
import warnings

warnings.filterwarnings('ignore')


def save_features(model, data_loader, outfile):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader)*data_loader.batch_size
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats=None
    count=0
    
    with torch.no_grad():
        for i, (x,y) in enumerate(data_loader):
            if i%50 == 0:
                print('{:d}/{:d}'.format(i, len(data_loader)))
            x = x.cuda()
            x_var = Variable(x)
            feats = model.forward(x_var)
            if all_feats is None:
                all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
            all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
            all_labels[count:count+feats.size(0)] = y.cpu().numpy()
            count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count
    
    f.close()


if __name__ == '__main__':
    params = parse_args('save_features')
    print(params)

    image_size = 224
    # if params.dataset == 'CUB':
    #     image_size = 224
    # elif params.dataset == 'miniImagenet':
    #     image_size = 84
    print('image_size = ', image_size)
    base_file = os.path.join('./filelists', params.dataset, 'base.json')
    val_file = os.path.join('./filelists', params.dataset, 'val.json')
    novel_file = os.path.join('./filelists', params.dataset, 'novel.json')
    # loadfile_list = [configs.data_dir[params.dataset] + 'base.json', configs.data_dir[params.dataset] + 'val.json', configs.data_dir[params.dataset] + 'novel.json']
    loadfile_list = [base_file, val_file, novel_file]
    split_list = ['base', 'val', 'novel']
    # loadfile = configs.data_dir[params.dataset] + split + '.json'
    params.checkpoint_dir = 'checkpoints/%s/%s_%s' %(params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'

    params.checkpoint_dir += '_%s_lr%s_%s_wd%s' % (params.optim, str(params.init_lr), params.lr_anneal, str(params.wd))

    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
    params.model_dir = os.path.join(params.checkpoint_dir, 'model')

    model = model_dict[params.model]()    # resnet10
    model = model.cuda()
    print("model_dir = ", params.model_dir)
    # load pre-trained model
    modelfile   = get_best_file(params.model_dir)
    print("modelfile = ", modelfile)
    tmp = torch.load(modelfile)
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
            state[newkey] = state.pop(key)
        else:
            state.pop(key)
    
    model.load_state_dict(state)
    model.eval()
    
    for i, loadfile in enumerate(loadfile_list):   # base, val, novel
        
        outfile = os.path.join(params.checkpoint_dir.replace("checkpoints","features"), split_list[i] + '_best' + ".hdf5")  # './features/miniImagenet/Conv4_baseline_aug/novel.hdf5'
        print('outfile = ', outfile)
        datamgr         = SimpleDataManager(image_size, batch_size = 64)

        data_loader      = datamgr.get_data_loader(loadfile, aug = False)

        dirname = os.path.dirname(outfile)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        print('begin save feature in ', split_list[i])
        save_features(model, data_loader, outfile)
