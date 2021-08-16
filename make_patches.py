import numpy as np
import pandas as pd
import cv2 as cv
import os
from typing import Union
from torchvision.utils import save_image
import dataloader.data_loaders as dl
from PIL import Image
import json
import torch

def transform_img(img: torch.Tensor, targetMean: torch.Tensor, targetSTD: torch.Tensor) -> torch.Tensor:
    """
    Implementation of color normalization method from Reinhard et al.
    @param img: Image (as a torch tensor) that will be transformed to match the target image's color distribution
    @param targetMean: Tensor of mean pixel activations for the RGB layers of target image
    @param targetSTD: Tensor of pixel standard deviations for RGB layers of target images
    """

    # Collect means and standard deviations from RGB channels
    imgMean, imgStd = img.mean((1,2)), img.std((1,2))
    meanRed, meanGreen, meanBlue = imgMean.tolist()
    stdRed, stdGreen, stdBlue = imgStd.tolist()
    targetMeanRed, targetMeanGreen, targetMeanBlue = targetMean.tolist()
    targetStdRed, targetStdGreen, targetStdBlue = targetSTD.tolist()
    
    # Reinhard transformation
    img[0,:,:] = ((img[0,:,:] - meanRed) / stdRed) * targetStdRed + targetMeanRed
    img[1,:,:] = ((img[1,:,:] - meanGreen) / stdGreen) * targetStdGreen + targetMeanGreen
    img[2,:,:] = ((img[2,:,:] - meanBlue) / stdBlue) * targetStdBlue + targetMeanBlue
    
    return img

def make_patches(img: torch.Tensor, label: torch.Tensor, win_size: Union[tuple,list], training: bool = True, 
                save_dir: str = 'data/patches', overlap: float = 0.5, test: bool = False) -> None:
    """
    Takes an (image, label) pair as input, divides the image into patches, and saves the patches with all corresponding 
    bounding box predictions into the jsonl format required for the AutoML tool.
    """
    json_line_sample = {
        'imageUrl': '',
        'imageDetails': {'format':'tiff', 'width':'', 'height':''},
        'label': {}
    }
    classes = ['not_mitotic', 'mitotic'] # Labels for bounding boxes
    h, w = img.size()[1], img.size()[2]
    name = img_names[label['image_id']][:-5] # last 5 characters are .tiff
    # Start cropping
    winX, winY = win_size[0], win_size[1] # Window height and width
    topX, topY = 0, 0 # Starting from top left corner
    
    with open(train_file if training else validation_file, 'a') as f:
        for y in range(h // int(winY*overlap)): # Shift window up-down taking into account overlap
            for x in range(w // int(winX*overlap)): # Slide window from left to right 
                save_path = os.path.join(save_dir, f'{name}_{topX}_{topY}.tiff')
                bottomX = topX + winX
                bottomY = topY + winY
                crop = img[:, topY:bottomY, topX:bottomX]
                # Check if there are any bboxes in this crop
                for i, box in enumerate(label['boxes']):
                    if box[0].item() in range(topX, bottomX):
                        if box[1].item() in range(topY, bottomY): # Check top left coordinate of box is in this crop
                            # Subtract original box coordinates from patch's origin point
                            c_x1 = box[0] - topX
                            c_y1 = box[1] - topY
                            # Need to check if bottom (x,y) coordinates are outside of the patch
                            c_x2 = (box[2] - topX) if box[2] in range(topX, bottomX) else bottomX
                            c_y2 = (box[3] - topY) if box[3] in range(topY, bottomY) else bottomY
                            # Check to make sure area of bounding box is > 40% original size 
                            # (original bboxes had areas of 4900px and any box less than 40% of original box area is removed)
                            h, w = crop.size()[1], crop.size()[2]
                            c_x2 = min(w, c_x2)
                            c_y2 = min(h, c_y2)
                            box_width = c_x2 - c_x1
                            box_height = c_y2 - c_y1
                            box_area = box_height * box_width
                            if box_area < (0.4*4900): 
                                continue
                            
                            labels = {}
                            labels['label'] = classes[label['labels'][i].item()]
                            # If (x,y) coordinates lie outside of our crop we reset them to be 1.0 for the autoML script
                            labels['topX'] = min(1.0, float(c_x1 / crop.size()[2]))
                            labels['topY'] = min(1.0, float(c_y1 / crop.size()[1]))
                            labels['bottomX'] = min(1.0, float(c_x2 / crop.size()[2]))
                            labels['bottomY'] = min(1.0, float(c_y2 / crop.size()[1]))
                            labels['isCrowd'] = 'false'
                            labels['isTruncated'] = 'false'
                            labels['mask'] = ''     
                            labels['area'] = box_area.item()
                            
                            json_line = dict(json_line_sample)
                            json_line['imageUrl'] = save_path
                            json_line['imageDetails']['height'] = str(crop.size()[1]) + 'px'
                            json_line['imageDetails']['width'] = str(crop.size()[2]) + 'px'
                            json_line['label'] = labels
                            f.write(json.dumps(json_line) + '\n')

                save_image(tensor=crop, fp=save_path)
                topX += int(winX * overlap) 

            # Reset X coordinate and shift down one row
            topY += int(winY * overlap)
            topX = 0

if __name__ == '__main__':
    batch_size = 16
    train_loader = dl.MitosisDataLoader('data', batch_size=batch_size, shuffle=False, validation_split=0.2, training=True)
    val_loader = train_loader.split_validation()
    root = 'data/images'
    img_names = list(sorted(os.listdir(root)))
    patch_window_size = (256,256)
    train_file = 'ICPR2014_patches_256_256_reinhard_train.jsonl'
    validation_file = 'ICPR2014_patches_256_256_reinhard_val.jsonl'
    save_dir = 'data/patches_256_256_reinhard'
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    targetImg = train_loader.dataset[10][0]
    targetM, targetS = targetImg.mean((1,2)), targetImg.std((1,2))
    print(f'saving patches to {save_dir}...')
    for i, loader in enumerate([val_loader, train_loader]):
        for imgs, labels in loader:   
            for i in range(batch_size):
                try:
                    image = imgs[i]
                    img = transform_img(img=image, targetMean=targetM, targetSTD=targetS)
                    make_patches(img=image, label=labels[i], win_size=patch_window_size, 
                                training=i, save_dir=save_dir, overlap=0.5)
                except: 
                    continue
            