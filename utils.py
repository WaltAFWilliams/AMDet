import numpy as np
import cv2 as cv
import torch, torchvision
import os
import matplotlib.pyplot as plt

def get_model(path_to_model_dict: str):
    model_dict = torch.load(path_to_model_dict)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features=1024, out_features=3, bias=True)
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features=1024, out_features=12, bias=True)
    model.to('cuda')
    model.eval()
    model.load_state_dict(model_dict['model_state'])
    return model


def drawPredictions(originalImg: torch.Tensor, patches: torch.Tensor, origins: list, model, threshold: float = 0.3) -> torch.Tensor:
    # Gather predictions for all patches
    # Draw bounding box predictions on all patches
    # Place patches on top of original image
    torch.cuda.empty_cache()
    drawn_patches = []
    drawnImg = originalImg.permute(1,2,0)
    classes = ['', 'mitotic', 'non-mitotic']
    patches = patches.to('cuda')
    preds = model(patches) # Gather predictions for all patches

    for i, pred in enumerate(preds): # Draw bounding boxes with image coordinates around all patches
        img = np.ascontiguousarray(patches[i].cpu().permute(1,2,0))
        for j, box in enumerate(pred['boxes']):
            if pred['scores'][j] >= threshold:
                box_as_int = [int(coord) for coord in box.tolist()]
                topX, topY, bottomX, bottomY = box_as_int
                label = classes[pred['labels'][j]]
                color = (0,0,255) if label=='mitotic' else (0,255,0) # Green for non-mitotic, blue for mitotic cells
                cv.rectangle(img, (topX, topY), (bottomX, bottomY), color, 5)
        drawn_patches.append(img)

    # Place patches back onto original image
    for i, patch in enumerate(drawn_patches):
        origin_x, origin_y = origins[i]
        drawnImg[origin_y:origin_y+512, origin_x:origin_x+512, :] = torch.as_tensor(patch, dtype=torch.float64)
    
    return drawnImg.permute(2,0,1)

def make_patches(img: torch.Tensor, win_size: tuple = (512, 512), overlap: float = 0.5) -> tuple:
    crops, origins = [], []
    slideDistance = 1.0 - overlap
    h, w = img.size()[1], img.size()[2]
    winX, winY = win_size[0], win_size[1] # Window height and width
    topX, topY = 0, 0 # Starting from top left corner

    # Start cropping
    for y in range(h // int(winY*slideDistance)): # Shift window up-down taking into account slideDistance
        for x in range(w // int(winX*slideDistance)): # Slide window from left to right 
            bottomX = topX + winX
            bottomY = topY + winY
            crop = img[:, topY:bottomY, topX:bottomX]
            if crop.size()[1] * crop.size()[2] == (winX*winY): 
                crops.append(crop)
                origins.append((topX, topY))
            topX += int(winX * slideDistance)
        
        topX = 0
        topY += int(winY * slideDistance)
    
    return (origins, torch.stack(crops))