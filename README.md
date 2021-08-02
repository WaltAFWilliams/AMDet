# AMDet: A Tool for Mitotic Cell Detection in Histopathology Slides.
By Walt Williams and James Hall at Microsoft Research

### Introduction
Welcome to the GitHub Repo for AMDet. This tool came as a result of a collaboration from 2 teams at Microsoft. There were pathologists working with the teams who were tasked with labeling a novel dataset for histopathology analysis but the process was incredibly slow and cumbersome according to them. In order to aid pathologists with the diagnosis of breast cancer in histopathology images, a tool for automatic localization and classification of mitotic and non-mitotic cells was needed. The original paper describing the details of the network can be found [here](www.google.com).

### How to Use
AMDet has 2 scripts in order to run detections. The first is `predict_on_one_image.py` which accepts a single image as an argument and will save the image (with all detections drawn) into a folder of your choosing. The second script, `predict_images.py` is virtually identical except you will pass in a folder with all the images you with to predict on inside of it. It will also save all predictions to a specified folder.

### Evaluating On One Image
Run this command replacing _path to folder_ with the path to the folder where the images are located and _path to saved model_ with the path to the model.pt file:
`Shell 
python predict_images.py -f <path to folder> -m <path to saved model>
`
The arguments for the `predict_image.py` script are:


