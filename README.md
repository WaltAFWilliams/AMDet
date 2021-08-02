# AMDet: A Tool for Mitotic Cell Detection in Histopathology Slides.
By Walt Williams and James Hall at Microsoft Research

### Introduction
Welcome to the GitHub Repo for AMDet. This tool came as a result of a collaboration from 2 teams at Microsoft. There were pathologists working with the teams who were tasked with labeling a novel dataset for histopathology analysis but the process was incredibly slow and cumbersome according to them. In order to aid pathologists with the diagnosis of breast cancer in histopathology images, a tool for automatic localization and classification of mitotic and non-mitotic cells was needed. The original paper describing the details of the network can be found [here](www.google.com).

### How to Use
AMDet has 2 scripts in order to run detections. The first is `predict_on_one_image.py` which accepts a single image as an argument and will save the image (with all detections drawn) into a folder of your choosing. The second script, `predict\_images.py` is virtually identical except you will pass in a folder with all the images you with to predict on inside of it. It will also save all predictions to a specified folder.
