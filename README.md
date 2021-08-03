# AMDet: A Tool for Mitotic Cell Detection in Histopathology Slides.
By Walt Williams and Jimmy Hall at Microsoft Research

### Introduction
Welcome to the GitHub Repo for AMDet. This tool came as a result of a need to help pathologists with labeling mitotic vs non-mitotic cells. There were pathologists who were tasked with labeling a novel dataset for histopathology analysis but they reported the process being incredibly slow and cumbersome. In order to aid them with their diagnoses a tool for automatic localization and classification of mitotic and non-mitotic cells was needed. The original paper describing the details of the network can be found [here](https://www.google.com).

### How To Use
AMDet is designed to run on histopathology slides roughly the size of the images in the [ICPR 2014 dataset](https://mitos-atypia-14.grand-challenge.org/Dataset/). It uses 2 scripts in order to run its detections. The first is `predict_on_one_image.py` which accepts a single image as an argument and will save the image (with all detections drawn) into a folder of your choosing. The second script, `predict_images.py` is virtually identical except you will pass in a folder with all the images you wish to pass through the network. It will also save all predictions to a specified folder.

### Evaluating On Multiple Images
Run this command replacing "_path to folder_" with the path to the folder where the images are located and "_path to saved model_" with the path to the model.pt file:
```Shell 
python predict_images.py -f <path to folder> -m <path to saved model>
```
The arguments for the `predict_images.py` script are:
```Shell
-f, --folder: path to folder where the images are located
-m, --model: path to saved torch model (.pt) format
-s, -patch_size: patch size to cut the image into before passing into network (OPTIONAL)
-d, --destination: path to folder where output will be stored (OPTIONAL)
```
***NOTE***: We do ***NOT*** recommend changing the patch size.

### Evaluating On One Image
The process for evaluating on a single image is identical except that the folder argument is replaced with a single image argument. For evaluating on a single image use this command:
```Shell
python predict_on_one_image.py -i <path to image> -m <path to saved model>
```
Again the arguments for this script are:
```Shell
-i, --image: path to image
-m, --model: path to saved torch model (.pt) format
-s, -patch_size: patch size to cut the image into before passing into network (OPTIONAL)
-d, --destination: path to folder where output will be stored (OPTIONAL)
```

### Acknowledgement
Special thanks to Jimmy Hall and his help with this project. It wouldn't have been possible without his mentorship and guidance.