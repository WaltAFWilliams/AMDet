"""
Grab one image and split it into patches
Feed all patches through model for prediction
Draw bounding boxes on top of all patches
Place patches back onto original image
Save drawn img into prediction directory
"""
import time
import numpy as np
import pandas as pd
import torch, torchvision
import matplotlib.pyplot as plt
import os
import cv2 as cv
from utils import *
import argparse

parser = argparse.ArgumentParser(description='Make A prediction on one image. The output will be saved to a directory: \"predictions\" unless otherwise specified.')
parser.add_argument('-i','--image', help='Path to image for prediction', required=True)
parser.add_argument('-m', '--model', help='Path to saved torch model (.pt) format', required=True)
parser.add_argument('-s', '--patch_size', type=int, 
                    help='Patch size to cut the image into before passing it into network for predictions. Default is 512.', default=512, required=False)
parser.add_argument('-d', '--destination', help='Folder where output will be stored.', default='mitosis_predictions', required=False)
args = vars(parser.parse_args())

if __name__=='__main__':
    imgPath = args['image']
    modelPath = args['model']
    patchSize = (args['patch_size'], args['patch_size']) 
    saveDir = args['destination']
    if not os.path.isdir(saveDir): os.mkdir(saveDir)
    T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    img = T(cv.cvtColor(cv.imread(imgPath), cv.COLOR_BGR2RGB))
    origins, patches = make_patches(img=img, win_size=patchSize, overlap=0.25)

    start = time.time()
    print('Fetching Model...')
    model = get_model(modelPath)

    print('Making prediction...')
    output = drawPredictions(originalImg=img, patches=patches, origins=origins, model=model, threshold=0.5)

    savePath = os.path.join(saveDir, imgPath.split('/')[-1])
    print(f'Saving prediction to {savePath}')
    torchvision.utils.save_image(tensor=output, fp=savePath)
    end = time.time()
    print(f'finished in {end-start:.1f}s')

