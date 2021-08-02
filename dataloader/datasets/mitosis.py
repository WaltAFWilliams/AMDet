from __future__ import print_function, division
import os
from os import listdir
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import transforms as T
from dataloader import custom_trans as tr
import sys
import tarfile
import collections
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg
import torch



class MitosisDataset(Dataset):
    def __init__(self, path, transforms = None):
        self.path = path
        self.transforms = transforms
        # load all image files, sorting them to
        super().__init__()
        self.imgdir = os.path.join(path, 'images')
        self.labeldir = os.path.join(path, 'labels')
        self.imgs = list(sorted(os.listdir(self.imgdir)))
        self.labels = list(sorted(os.listdir(self.labeldir)))
        # print(len(self.imgs))
        # print(len(self.labels))
        for i in self.labels:
            if not i.endswith('csv'):
                print(i)
        assert len(self.imgs) * 2 == len(self.labels)
    
        to_remove = []
        for i, img in enumerate(self.imgs):
            
            if self._annotation_empty(img, mitosis=True) and self._annotation_empty(img, mitosis=False):
                to_remove.append(i)
                #print("skipping ", i, img)
                # 668 40x tiles no not_mit or mitotic event
        # print(len(to_remove))
        self.imgs = [img for i,img in enumerate(self.imgs) if i not in to_remove]


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # load images and masks
        
        not_mit = self._read_csv(self._get_path(self.imgs[idx], mitosis=False)[1])
        mit = self._read_csv(self._get_path(self.imgs[idx], mitosis=True)[1])
        img_path = self._get_path(self.imgs[idx])[0]
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        
        size = 35 # TODO: Manually defined here, every box will be 70 by 70 size 
        for i,annot in enumerate([not_mit, mit]):
            for m in annot:
                center_x, center_y, score = m
                x0 = center_x - size 
                y0 = center_y - size
                x1 = center_x + size
                y1 = center_y + size
                boxes.append((x0,y0,x1,y1))
                labels.append(i)
        
        # convert everything into a torch.Tensor
        #import pdb; pdb.set_trace()
        boxes = torch.from_numpy(np.array(boxes, dtype=int))
        labels = torch.from_numpy(np.array(labels, dtype=int)).type(torch.int64)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # print(boxes)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64) # only needed to make eval code work
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def _read_csv(self, filename):
        annot = []
        with open(filename, 'r') as f:
            for line in f:
                x,y,score = line.strip('\n').split(',')
                annot.append((int(x),int(y), float(score)))
        return annot

    def _get_path(self, img, mitosis=False):
        img_path = os.path.join(self.imgdir, img)
        if not mitosis:
            ext = "_not_mitosis.csv"
        else:
            ext = "_mitosis.csv"
            
        label_path = os.path.join(self.labeldir, img.replace('.tiff', ext))
        return img_path, label_path
        
    def _annotation_empty(self, img, mitosis=False):
        return len(self._read_csv(self._get_path(img, mitosis=mitosis)[1])) == 0