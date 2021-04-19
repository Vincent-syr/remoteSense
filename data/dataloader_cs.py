import torch
from PIL import Image
import numpy as np
from torchvision import transforms, datasets

# import data.additional_transforms as add_transforms
# from data.dataset_npy import MiniImgDataset
from abc import abstractmethod   
    


    
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def get_loader():
    data_folder = "/test2/0Dataset_others/Dataset/RemoteSense/NWPU-RESISC45/NWPU-RESISC45"

    mean= (0.485, 0.456, 0.406) 
    std=(0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])


    train_dataset = datasets.ImageFolder(root=data_folder,
                                            transform=TwoCropTransform(train_transform))
    
    print(len(train_dataset[0][0]))   # 2
    print(train_dataset[0][0][0].shape)   # torch.Size([3, 256, 256])


    # sampler = EpisodicSampler(dataset.label, self.n_way, self.batch_size, self.n_episode)



if __name__ == '__main__':
    get_loader()