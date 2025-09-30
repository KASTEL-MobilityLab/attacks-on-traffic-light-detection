import cv2
import os
import sys
sys.path.append("./yolov7")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.patches as patches
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

# yolov7 imports
from models.experimental import attempt_load
from utils.loss import ComputeLoss
from utils.datasets import LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, TracedModel

from patch_utils.general import load_classes


def run_inference_for_one_image(img_for_inference, img_vor_viz, target_file, rgb_image = True):
    model.eval()
    
    plt.rcParams['figure.figsize'] = [100, 100]

    # Inference
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img_for_inference, augment=False)[0]

    # Apply NMS
    print("Predictions ", len(pred[0]))
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, agnostic=True)
    print("Predictions after NMS ", len(pred[0]), pred)

    # Process detections
    for i, det in enumerate(pred):
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img_vor_viz, label=label, color=colors[int(cls)], line_thickness=2)

        # Show results
        if not rgb_image:
            rgb_img = cv2.cvtColor(img_vor_viz, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = img_vor_viz
#         if rgb_img.shape[0]==rgb_img.shape[1]:
#             rgb_img = rgb_img[128:512,:,:]
        fig = plt.imshow(rgb_img)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        
        plt.savefig(target_file)
        

        
if __name__ == "__main__":
    
    IOU_THRES = 0.5
    CONF_THRES = 0.5
    IMG_SIZE = 1280

    yolo_weights_path = "./weights/dtld/dtld_ccng_1280/weights/best.pt"
    classes_path = "configs/dtld/classes.names"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    half = device.type != 'cpu'
    print(device)
    print(half)
    half = False

    model = attempt_load(yolo_weights_path, map_location=device) 

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Set a fixed seed for the random number generator
    random.seed(21)  # You can choose any number as a seed

    # Function to generate a random color
    def random_color():
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    # Generate a list of colors for each class
    bstld_colors = [(128,0,0), (128,128,0), (0,128,0), (0,0,128)]
    lisa_colors = [(128,0,0), (0,128,0), (0,128,0), (128,0,0), (128,128,0), (128,128,0), (0,128,0)]
    colors = [random_color() for _ in range(len(names))]
    #colors = lisa_colors

    colors[0] = (0,128,0)
    colors[1] = (128,0,0)
    print(names)

    classes = load_classes(classes_path)
    print(classes)
    
    img_path = "./"
    target_path = "./"


    for path, subdirs, files in os.walk(img_path):
        for img_name in files:
            full_img_name = img_path + img_name
            print(full_img_name)
            img0 = cv2.imread(full_img_name)
            img_letterbox = letterbox(img0, IMG_SIZE, stride=32)[0]

            img_to_rgb = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)

            # Convert
            img = img_letterbox[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            target_file = target_path + img_name
            run_inference_for_one_image(img, img_to_rgb, target_file)