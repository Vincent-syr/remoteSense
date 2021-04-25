# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

identity = lambda x:x
class SimpleDataset:
    def __init__(self, data_file, transform, params, aug, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.data = self.meta['image_names']
        self.label = self.meta['image_labels']
        self.params = params
        self.aug = aug
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img1 = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        if ('cs' in self.params.method)and self.aug:
            img2 = self.transform(img)
            return (img1, img2), target    # aug for contrastive learning
        else:
            return img1, target

    def __len__(self):
        return len(self.meta['image_names'])


class EpisodicSampler(object):
    def __init__(self, label, n_way, n_per, n_episodes):   
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.n_per = n_per
        label = np.array(label)
        self.m_ind = []
        
        for i in np.unique(label):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)     

    
    def __len__(self):
        return self.n_episodes


    def __iter__(self):
        """ iterater of each episode
        """
        for i_batch in range(self.n_episodes):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_way]
            for c in classes:
                l = self.m_ind[c]  # all samples of c class
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            # batch = torch.stack(batch).t().reshape(-1)
            batch = torch.stack(batch).reshape(-1)
            yield batch