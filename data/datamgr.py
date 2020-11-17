import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SimpleDataset, EpisodicSampler
# from data.dataset_npy import MiniImgDataset
from abc import abstractmethod


class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomSizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Scale':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 


class SetDataManager(DataManager):
    def __init__(self, image_size, n_way, n_support, n_query, params, aux=False,  n_episode =100):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode
        self.trans_loader = TransformLoader(image_size)
        self.params = params

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(data_file, transform)
        sampler = EpisodicSampler(dataset.label, self.n_way, self.batch_size, self.n_episode)
        # data_loader_params = dict(batch_sampler = sampler,  num_workers = 12, pin_memory = True)       
        # data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        if self.params.os == 'linux':
            # print('os = linux')
            data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=12, pin_memory=True)
        else:
            data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)

        return data_loader
