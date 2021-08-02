from torchvision import datasets, transforms
from base import BaseDataLoader
from dataloader.datasets import *
import dataloader.custom_trans as T
import torchvision.transforms as trans



def collate_fn(batch):
    return tuple(zip(*batch))

class MitosisDataLoader(BaseDataLoader):
    """
    Atypia ICPR2014 data loader for pytorch
    """
    
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=6, training=True):
        self.data_dir = data_dir
        
        tfs = self.get_transform(training)
        self.dataset = MitosisDataset(self.data_dir, transforms=tfs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn = collate_fn)
        
    def get_transform(self, train):
        transforms = []
        transforms.append(T.ToTensor())
        # transforms.append(T.Resize())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)