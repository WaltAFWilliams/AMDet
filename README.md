# AMDet: A Tool for Mitotic Cell Detection in Histopathology Slides
By Walt Williams and Jimmy Hall at Microsoft Research

## Introduction
This work was inspired by a need to help a group of pathologists with labeling a novel dataset for histopathology analysis. They reported the process as being incredibly slow and cumbersome and so in order to aid them with their diagnoses a tool for automatic localization and classification of mitotic and non-mitotic cells was needed, thus giving rise to AMDet. For complete information about AMDet and the process of creating it please refer to the original paper describing the details of the network [here](https://arxiv.org/abs/2108.03676). In short, AMDet uses 2 scripts in order to run its detections. The first is `predict_on_one_image.py` which runs predictions on a single image then saves the image (with all detections drawn) into a folder of your choosing. The second script, `predict_images.py` is virtually identical except that it runs detections on multiple images and you have to specify a folder containing all the images you wish to pass through the network. It also saves all predictions to a specified folder.

## How To Use
### Clone Repo
First, clone the repo and switch to it with the following commands:
```Shell
git clone https://github.com/WaltAFWilliams/AMDet.git
cd AMDet
```

### Download Weights
Download the model's weights using [this link](http://aka.ms/automl-research-resources/data/models-vision-pretrained/amdet.pt) (NOTE: _the link does not work with google chrome_). Make sure to remember the location of this file as it's needed when you run the commands for getting the model's predictions. 

### Download Data
AMDet is designed to run on histopathology slides with dimensions roughly equal to those found in the [ICPR 2014 dataset](https://mitos-atypia-14.grand-challenge.org/Dataset/). If you do not have your own slides and would like to run sample detections using AMDet you can download a set of images from the ICPR 2014 competition [here](https://mega.nz/folder/uRpm2AZI#B_vY4ZZw_eIUpbFV1sEqKA). When the download is finished simply extract the images to a folder and continue to the next step.

### Install Dependencies
For smooth use of AMDet we recommend installing the dependencies via pip:
```Shell
pip install -r requirements.txt
```

### Evaluating On Multiple Images
Run this command replacing "_path to folder_" with the path to the folder where the images are located and "_path to model weights_" with the path to the model's weights':
```Shell 
python3 predict_images.py -f <path to folder> -m <path to model weights>
```
The arguments for the `predict_images.py` script are:
```Shell
-f, --folder: path to folder where the images are located
-m, --model: path to saved torch model (.pt) format
-s, --patch_size: patch size to cut the image into before passing into network (OPTIONAL)
-d, --destination: path to folder where output will be stored (OPTIONAL)
```
***NOTE***: We do ***NOT*** recommend changing the patch size.

### Evaluating On One Image
The process for evaluating on a single image is identical except that the folder argument is replaced with a single image argument. For evaluating on a single image use this command:
```Shell
python3 predict_on_one_image.py -i <path to image> -m <path to saved model>
```
The arguments for this script are:
```Shell
-i, --image: path to image
-m, --model: path to saved torch model (.pt) format
-s, --patch_size: patch size to cut the image into before passing into network (OPTIONAL)
-d, --destination: path to folder where output will be stored (OPTIONAL)
```

### Acknowledgement
Special thanks to Jimmy Hall and his help with this project. It wouldn't have been possible without his mentorship and guidance.
