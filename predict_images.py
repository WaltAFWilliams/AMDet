"""
Similar to 'predict_one_one_image.py' except predicts on all images in a folder instead.
"""

import numpy as np
import pandas as pd
import torch, torchvision
import matplotlib.pyplot as plt
import os
import cv2 as cv
from utils import make_patches, drawPredictions, get_model
import argparse
import time

parser = argparse.ArgumentParser(description='Make predictions on all images in a folder. The output will be saved to a directory: \"mitosis_predictions\" unless otherwise specified.')
parser.add_argument('-f','--folder', help='Path to folder containing images for prediction', required=True)
parser.add_argument('-m', '--model', help='Path to saved torch model (.pt) format', required=True)
parser.add_argument('-s', '--patch_size', type=int, 
                    help='Patch size to cut the image into before passing it into network for predictions. Default is 512', default=512, required=False)
parser.add_argument('-d', '--destination', help='Folder where output will be stored.', default='mitosis_predictions', required=False)
args = vars(parser.parse_args())

if __name__=='__main__':
    folderRoot = args['folder']
    imgPaths = [os.path.join(folderRoot, path) for path in os.listdir(folderRoot)]
    modelPath = args['model']
    patchSize = (args['patch_size'], args['patch_size']) 
    saveDir = args['destination']
    if not os.path.isdir(saveDir): os.mkdir(saveDir) # Create directory if not already made
    T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    
    print('Fetching model...')
    model = get_model(modelPath)
    
    print('Making predictions...')
    start = time.time()
    for imgPath in imgPaths:
        img = T(cv.cvtColor(cv.imread(imgPath), cv.COLOR_BGR2RGB))
        origins, patches = make_patches(img=img, win_size=patchSize, overlap=0.5)
        output = drawPredictions(originalImg=img, patches=patches, origins=origins, model=model, threshold=0.5)
        savePath = os.path.join(saveDir, imgPath.split('/')[-1])
        torchvision.utils.save_image(tensor=output, fp=savePath, format='png')
    
    end = time.time()
    print(f'Finished in {(end-start):.1f}s.\nAll predictions saved to folder: \"./{saveDir}\"')

