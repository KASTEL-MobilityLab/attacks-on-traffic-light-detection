"""
This script will train a Projected Gradient Descent adversarial patch
on a dataset.

Run it with a config file. Examples found in /configs
E.g.
$ python train_patch_yolov8.py configs/config_patch_dtld_yolov8.yaml
"""
import cv2
from datetime import datetime
from tqdm import tqdm
import yaml
import platform
import wandb

import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.data import YOLODataset
from ultralytics.cfg import get_cfg


# from patch_utils.datasets import load_dataset
from patch_utils.losses import yolo_loss, total_variation_loss, v8AttackLoss
from patch_utils.transforms import patch_brightness, patch_pad, patch_rotate, patch_resize
from ultralytics.utils.loss import v8DetectionLoss

# Patch imports
from patch_utils.general import load_classes, store_wandb_image_with_bboxes
from patch_utils.losses import yolo_loss, total_variation_loss, green_channel_penalty, similarity_loss
from patch_utils.transforms import patch_brightness, patch_pad, patch_rotate, patch_resize


def transform_patch(current_patch):
    """
    Transform patch slightly at each PGD step to improve
    real world attack effectiveness.
    """
    
    # Adjust patch brightness
    brightness = np.random.uniform(0.4, 1.6)
    current_patch = patch_brightness(current_patch, brightness)

    # Pad patch
    pad_width = 5  # space around patch preventing it from getting cut off after rotation
    current_patch = patch_pad(current_patch, pad_width)

    # Create mask
    mask = torch.ones_like(current_patch)

    # Rotate patch and mask
    anglex = np.random.uniform(-5, 5)
    angley = np.random.uniform(-5, 5)
    anglez = np.random.uniform(-10, 10)
    current_patch = patch_rotate(current_patch, anglex, angley, anglez)
    mask = patch_rotate(mask, anglex, angley, anglez)

    return current_patch, mask


def load_initial_patch(image_path, patch_size):
    """
    Used to convert an image to a patch when training
    a patch starting from an image.
    """
    
    image = Image.open(image_path).convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0  # Convert image to NumPy array and normalize
    patch = torch.from_numpy(image).permute(2, 0, 1)  # Convert to tensor and change to (C, H, W) format
    patch = patch_resize(patch, (patch_size, patch_size))  # Resize the patch to the desired size
    return patch.squeeze(0)


# Train the patch
def run_attack(params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = YOLO(params["yolo_weights_path"])
    model.to(device)
    classes = model.names
    
    img_size = params["img_size"]

    train_dataset = YOLODataset(params["train_img_folder_path"], 
                                imgsz=img_size, 
                                augment=False, 
                                batch_size=1, 
                                data=dict(names=list(range(1000))))
    val_dataset = YOLODataset(params["val_img_folder_path"], 
                                imgsz=img_size, 
                                augment=False, 
                                batch_size=1, 
                                data=dict(names=list(range(1000))))
    if params["second_val_dataset"]:
        val_dataset2 = YOLODataset(params["val_img_folder_path2"], 
                                imgsz=img_size, 
                                augment=False, 
                                batch_size=1, 
                                data=dict(names=list(range(1000))))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    if params["second_val_dataset"]:
        val_loader2 = DataLoader(val_dataset2, batch_size=1, shuffle=False)
    nb = len(train_loader)  # number of batches

    # Initialise patch
    delta = torch.empty(3, params["patch_size"], params["patch_size"]).uniform_(0, 1).to(device)

    # Initialise optimiser and learning rate scheduler
    optimizer = optim.Adam([delta], lr=params["learning_rate"], amsgrad=False)
    optimizer.zero_grad()
    
    model.model.train()  # set the model in the training mode
#     model.model.args["box"] = params["box_coef"] # BBox loss component
#     model.model.args["cls"] = params["cls_coef"] # Class loss component
#     model.model.args["dfl"] = params["dfl_coef"] # Distribution Focal Los component
    model.model.args["imgsz"] = img_size
    
    model.model.args = get_cfg(model.model.args, None)
    v8loss = v8AttackLoss(model.model)
    
    step = 0
    saved_patch_i = 0

    for epoch in tqdm(range(params["num_epochs"]), desc=f"Training"):
        epoch_loss = 0.0
        delta.requires_grad = True

        for batch in tqdm(train_loader):
            img = batch["img"]
            img = img.to(device)
            img = img.float() / 255.0
            cls = batch["cls"]
            bbox_coords = batch["bboxes"][0] #xywh
            
            # Replace IDs of class to attack with a target class in the ground-truth label
            tl_indices_to_attack = list()
            if not torch.numel(bbox_coords): # no BBoxes
                continue
            if params["only_lowest_bbox"]:
                y_coords = bbox_coords[:, 1]
                lowest_idx = torch.argmax(y_coords)
                tl_indices_to_attack = [lowest_idx]
            else:
                tl_indices_to_attack = range(len(bbox_coords))
            
            found_required_class = False
            for i in tl_indices_to_attack:
                if cls[:,i,:] == torch.tensor(params["class_to_replace"]):
                    found_required_class = True
            if not found_required_class:
                continue
      
            new_cls = cls.clone()
            patch_coords = list()
            for bbox_idx in tl_indices_to_attack:
                if cls[:,bbox_idx,:] != torch.tensor(params["class_to_replace"]):
                    continue
                new_cls[:,bbox_idx,:] = torch.tensor(params["target_class"]) # replace class 1 (red) with class 0 (green)
                
                # identify patch position and size
                current_bbox = bbox_coords[bbox_idx]
                x_center = current_bbox[0].cpu().detach().numpy()*img_size
                y_center = current_bbox[1].cpu().detach().numpy()*img_size
                bbox_width = current_bbox[2].cpu().detach().numpy()*img_size
                bbox_height = current_bbox[3].cpu().detach().numpy()*img_size

                patch_start_y = int(y_center + bbox_height/2) 
                patch_coords.append((x_center, bbox_width, patch_start_y))
            
            num_bboxes = cls.shape[1]
            
            # Start training a patch for 
            for _ in range(params["pgd_steps"]):
                adv_img = img.clone()

                for (x_center, bbox_width, patch_start_y) in patch_coords:
                    
                    current_patch_size = int(bbox_width*int(params["patch_width_multiplier"]))
                    current_patch_start_x = int(x_center - current_patch_size/2)
                    
                    # Transform patch
                    if params['transform']:
                        current_patch, _ = transform_patch(delta)   
                    else:
                        current_patch = delta
                    current_patch = patch_resize(current_patch, (current_patch_size, current_patch_size))
                    patch_end_x = current_patch_start_x + current_patch_size
                    patch_end_y = patch_start_y + current_patch_size

                    # If the patch is off the edge of the image
                    if patch_end_x>img_size or patch_end_y>img_size or current_patch_start_x<0 or patch_start_y<0:
                        continue

                    for i in range(img.shape[0]):
                        adv_img[i, :, patch_start_y : patch_end_y, current_patch_start_x : patch_end_x] = current_patch

                # Detection on adversarial image
                result_perturbed = model.model(adv_img)
                train_batch = {'cls': new_cls.to(device),
                               'bboxes': bbox_coords.to(device),
                               'batch_idx': torch.tensor([epoch]).repeat(num_bboxes).to(device)
                              }
                # YOLO loss
                yolo_loss = v8loss(result_perturbed, train_batch, )
                delta_batch = torch.unsqueeze(current_patch, dim=0)
                #delta_batch = torch.unsqueeze(delta, dim=0)
                tv_loss = total_variation_loss(delta_batch)
                loss = yolo_loss + params["tv_coef"]*tv_loss

                # Green penalty
#                 green_penalty = green_channel_penalty(delta)
#                 green_penalty = green_penalty * params.get("green_pen_coef", 0.001)

                # Calculate total loss
#                 total_loss = loss + tv_loss + green_penalty

                # Similarity loss
#                 if use_initial_image:
#                     sim_loss = similarity_loss(delta, initial_patch)
#                     sim_loss = sim_loss * params.get("sim_loss_coef", 1)
#                     total_loss += sim_loss
                
                epoch_loss += loss

                with torch.no_grad():
                    wandb.log({
                        "total_loss": loss.cpu().detach().numpy(),
                        "yolo_loss": yolo_loss.cpu().detach().numpy(),
                        "tv_loss": tv_loss.cpu().detach().numpy(),
#                         "green_penalty": green_penalty.cpu().detach().numpy(),
#                         "similarity_loss": sim_loss.cpu().detach().numpy() if use_initial_image else 0,
                    })

                loss.backward()  # Compute loss gradient

                optimizer.step()  # Update delta
                optimizer.zero_grad()  # Clear gradients

                delta.data = torch.clamp(delta.data, min=0, max=1) 
                
                step+=1
                
                if step%4==0:
                    save_image(delta, './'+str(saved_patch_i)+'.png')
                    saved_patch_i+=1

                
        mean_epoch_loss = epoch_loss / (params["pgd_steps"]*nb)
        wandb.log({"mean_epoch_loss": mean_epoch_loss,
                   "epoch": epoch+1})     

        # Save patch
        if epoch%params["save_freq"] == 0:
            model.model.eval()
            attack_effectiveness(params, delta, device, model, val_loader, "same_data", epoch%params["save_freq"])
            if params["second_val_dataset"]:
                attack_effectiveness(params, delta, device, model, val_loader2, "other_data", epoch%params["save_freq"])

            patch = delta.detach().cpu().numpy()
            patch = np.transpose(patch, (1, 2, 0))
            patch = wandb.Image(patch)
            wandb.log({"patch": patch})
            torch.save(delta.detach().cpu(), params["patch_name"] + str(epoch+1) + ".pt")
            
            # Set the model in the evaluation mode and run inference for the clean and perturbed images

            model.model.train()  # set the model in the training mode again
            



def attack_effectiveness(params, patch, device, model, loader, name, log_images):
    """
    Calculate a percentage score for how effective the attack is over a dataset.
    Based upon the classification of the highest confidence bounding box.

    NOTE: this function assumes that each image has exactly one ground truth label
    TODO: if to be used with datasets other than mobile, this needs fixed
    """
    
    total_imgs_tested = 0
    imgs_attacked = 0
    imgs_unattacked = 0
    imgs_vanishing=0
    patch_sizes = list()

    for batch in tqdm(loader, desc=f"Validation"):
        val_img = batch["img"]
        val_img = val_img.to(device)
        val_img = val_img.float() / 255.0
        cls = batch["cls"]

        bbox_coords = batch["bboxes"][0] #xywh
        y_coords = bbox_coords[:, 1]
        if not torch.numel(bbox_coords): # no BBoxes
            continue
        lowest_idx = torch.argmax(y_coords)
        lowest_bbox = bbox_coords[lowest_idx]
        tl_indices_to_attack = [lowest_idx]
        
        new_cls = cls.clone()
        
        if cls[:,lowest_idx,:] != torch.tensor(params["class_to_replace"]):
            continue
        
        
        # set patch to be bbox_width*2
        x_center = lowest_bbox[0].cpu().detach().numpy()*params["img_size"]
        y_center = lowest_bbox[1].cpu().detach().numpy()*params["img_size"]
        bbox_width = lowest_bbox[2].cpu().detach().numpy()*params["img_size"]
        bbox_height = lowest_bbox[3].cpu().detach().numpy()*params["img_size"]
        patch_size = int(bbox_width*int(params["patch_width_multiplier"]))
        patch_sizes.append(patch_size)
        

        patch_start_x = int(x_center - patch_size/2)
        patch_start_y = int(y_center + bbox_height/2) 
        patch_coords = list()
        patch_coords.append((x_center, bbox_width, patch_start_y))
        
        current_patch = patch_resize(patch,(patch_size, patch_size))
        patch_end_x = patch_start_x + patch_size
        patch_end_y = patch_start_y + patch_size
        
        if patch_end_x>params["img_size"] or patch_end_y>params["img_size"] or patch_start_x<0 or patch_start_y<0:
            continue
            
        img_to_store = val_img.clone()
        img_attacked_to_store = val_img.clone()
        
        for i in range(img_attacked_to_store.shape[0]):
            img_attacked_to_store[i, :, patch_start_y : patch_end_y, patch_start_x : patch_end_x] = current_patch
        
        with torch.no_grad():
            result_val = model.predict(img_attacked_to_store, save=False, conf=0.5, verbose=False)
            bboxes = result_val[0].boxes
            if bboxes.cls.shape[0]==0:
                imgs_vanishing +=1
            else:
                y_coords = bboxes.xywh[:, 1]
                lowest_idx = torch.argmax(y_coords)
                if bboxes.cls[lowest_idx] == torch.tensor(params["target_class"]):
                    imgs_attacked += 1
                else:
                    imgs_unattacked += 1

        total_imgs_tested += 1
        
#     print("bbox_coords", bbox_coords)
#     print("y_coords", y_coords)
#     print("lowest_idx", lowest_idx)
#     print("lowest_bbox", lowest_bbox)
    print("Patch sizes", patch_sizes)
    print("total_imgs_tested", total_imgs_tested)
    print("imgs_attacked", imgs_attacked)
    print("imgs_unattacked", imgs_unattacked)
    print("imgs_vanishing", imgs_vanishing)
        
    result_clean = model.predict(img_to_store, save=False, conf=0.5, verbose=False)
    annotated_img = result_clean[0].plot()
    annotated_img = annotated_img[:,:,::-1]
    annotated_img = wandb.Image(annotated_img)
    wandb.log({"Clean "+name: annotated_img})

    result_adv = model.predict(img_attacked_to_store, save=False, conf=0.5, verbose=False)
    annotated_img = result_adv[0].plot()
    annotated_img = annotated_img[:,:,::-1]
    annotated_img = wandb.Image(annotated_img)
    wandb.log({"Attacked "+name: annotated_img})
        
        
#         if len(out) > 0:
#             # If any red predictions made it past nms, the attack was unsuccessful
#             if any(int(x) == int(params.get("class_to_replace")) for x in out[:, 5]):
#                 still_had_red_bb += 1
        
#             # Find the prediction with the highest confidence
#             _, index = torch.max(out[:, 4], 0)
#             main_prediction = out[index]

#             if int(main_prediction[5]) == int(params.get("target_class")):
#                 images_successfully_attacked += 1
    
#     if log_images:
#         # Log an image to see the progress on it
#         bbox_adv_image = store_wandb_image_with_bboxes(img, [out], classes)
#         wandb.log({"Adversarial image (from val split)": bbox_adv_image})

#     print("Evaluated images:", total_imgs_tested)
#     print("Attacked:", imgs_attacked / total_imgs_tested)
#     print("Failed:", imgs_unattacked / total_imgs_tested)
#     print("Vanishing:", imgs_vanishing / total_imgs_tested)
    wandb.log({"Attack successful " + name: imgs_attacked/total_imgs_tested,
           "No attack effect " + name: imgs_unattacked/total_imgs_tested,
           "Vanishing object effect " + name: imgs_vanishing/total_imgs_tested})    


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('\nPlease pass the desired param file for training as an argument.\n'
              'e.g: configs/config_patch_green.yaml')
    else:
        config_file_path = str(sys.argv[1])
        print('STARTING PGD ATTACK WITH PARAM FILE: ', config_file_path)
        with open(config_file_path, 'r') as stream:
            try:
                ## initialize wandb
                wandb.init(project="tlr-attacks", reinit=True)
                
                config = yaml.safe_load(stream)
                wandb.config.update(config)
                wandb.config.update({"machine_name": platform.node()})
                
                ## initialize params from loaded config file
                params = wandb.config

                wandb.run.name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_universal_targeted_" + params["dataset"] + "_"+ str(params["img_size"])
                wandb.run.save()

                # Set random seeds
                random_seed = params.get("random_seed", 1)
                torch.manual_seed(random_seed)
                np.random.seed(random_seed)

                run_attack(params)

            except yaml.YAMLError as exc:
                print(exc)
