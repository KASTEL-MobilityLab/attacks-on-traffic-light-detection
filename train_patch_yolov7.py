"""
This script will train a Projected Gradient Descent adversarial patch
on a dataset.

Run it with a config file. Examples found in /configs
E.g.
$ python train_pgd_patch.py configs/config_patch_mobile.yaml
"""

from datetime import datetime
from tqdm import tqdm
import yaml
import platform
import wandb

import numpy as np
from PIL import Image

import torch
import torch.optim as optim

# yolov7 imports
import sys
sys.path.append("yolov7")
from models.experimental import attempt_load
from utils.general import non_max_suppression

# Patch imports
from patch_utils.general import load_classes, store_wandb_image_with_bboxes
from patch_utils.datasets import load_dataset
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

    model = attempt_load(params["yolo_weights_path"], map_location=device)
    classes = load_classes(params["classes_path"])

    train_dataloader = load_dataset(params["img_folder_path"], params["img_size"])
    val_dataloader = load_dataset(params["val_img_folder_path"], params["img_size"])

    # Fetch config variables
    pgd_steps = params.get("pgd_steps", 10) # Perform 10 pgd steps by default
    save_freq = params.get("save_freq", 10) # Save images and weights every 10 epochs by default

    # Initialise patch
    use_initial_image = params.get("use_initial_image", False)
    if use_initial_image:
        initial_patch = load_initial_patch(params.get("initial_image"), params["patch_size"]).to(device)
        delta = initial_patch.clone().detach().requires_grad_(True)
    else:
        delta = torch.empty(3, params["patch_size"], params["patch_size"]).uniform_(0, 1).to(device)

    # Initialise optimiser and learning rate scheduler
    optimizer = optim.Adam([delta], lr=params["learning_rate"], amsgrad=False)
    optimizer.zero_grad()

    for epoch in tqdm(range(params["num_epochs"]), desc=f"Training"):
        epoch_loss = 0.0
        delta.requires_grad = True

        for (img, gt_label, _, _) in train_dataloader:
            # Move current image and label to device
            img = img.to(device)
            img = img.float() / 255.0
            gt_label = gt_label.to(device)

            # Skip this image if there are no labels, or no labels of the right colour
            if len(gt_label) == 0:
                continue

            found_required_class = False
            for gtx in gt_label:
                if gtx[1] == int(params["class_to_replace"]):
                    found_required_class = True
            if not found_required_class:
                continue

            # Calculate coordinates to place patches
            # gt_label in format (class_id, x_centre,  y_centre,  width,  height)
            patch_coords = list()
            for gtx in gt_label:
                x_center = gtx[2].cpu().detach().numpy()*params["img_size"]
                y_center = gtx[3].cpu().detach().numpy()*params["img_size"]
                bbox_width = gtx[4].cpu().detach().numpy()*params["img_size"]
                bbox_height = gtx[5].cpu().detach().numpy()*params["img_size"]
                patch_start_y = int(y_center + bbox_height/2)
                patch_coords.append((x_center, bbox_width, patch_start_y))

                # replace labels for a targeted attack
                if gtx[1] == int(params["class_to_replace"]):
                    gtx[1] = int(params["target_class"])

            for _ in range(pgd_steps):
                adv_img = img.clone()

                # Transform patch
                if params['transform']:
                    current_patch, mask = transform_patch(delta)   
                else:
                    current_patch = delta
                    mask = torch.ones_like(delta)

                # Apply patch
                tl_bbox_found = False
                for (x_center, bbox_width, patch_start_y) in patch_coords:

                    # Resize patch to fit traffic light
                    transformed_patch_size = int(bbox_width * params.get("patch_width_multiplier", 2))
                    current_patch_resized = patch_resize(current_patch, (transformed_patch_size, transformed_patch_size))
                    mask_resized = patch_resize(mask, (transformed_patch_size, transformed_patch_size))

                    # Calculate patch coordinates
                    patch_start_x = int(x_center - transformed_patch_size/2)
                    patch_end_x = patch_start_x + transformed_patch_size
                    patch_start_y = patch_start_y + np.random.randint(0, int(bbox_width / 2)) # Randomly vary y position
                    patch_end_y = patch_start_y + transformed_patch_size

                    # If the patch is off the edge of the image
                    if patch_end_x > params["img_size"] or patch_end_y > params["img_size"] or patch_start_x < 0 or patch_start_y < 0:
                        tl_bbox_found = False
                        continue
                    tl_bbox_found = True

                    # Patch application
                    for i in range(img.shape[0]):                
                        adv_img[i, :, patch_start_y:patch_end_y, patch_start_x:patch_end_x] = \
                            adv_img[i, :, patch_start_y:patch_end_y, patch_start_x:patch_end_x] * (1 - mask_resized) + \
                            current_patch_resized * mask_resized

                if tl_bbox_found:
                    prev_adv_img = adv_img.clone()
                    prev_img = img.clone()     
                          
                # Detection on adversarial image
                _, preds_parts = model(adv_img)
                preds = [x.float() for x in preds_parts]
                
                # YOLO loss
                lbox, lcls = yolo_loss(model, preds, gt_label, device)
                lbox = lbox * params.get("box_coef", 1)
                lcls = lcls * params.get("cls_coef", 1)
                
                # Total variation loss 
                delta_batch = torch.unsqueeze(delta, dim=0)
                tv_loss = total_variation_loss(delta_batch)
                tv_loss = tv_loss * params.get("tv_coef", 1)

                # Green penalty
                green_penalty = green_channel_penalty(delta)
                green_penalty = green_penalty * params.get("green_pen_coef", 0.001)

                # Calculate total loss
                total_loss = lcls + lbox + tv_loss + green_penalty

                # Similarity loss
                if use_initial_image:
                    sim_loss = similarity_loss(delta, initial_patch)
                    sim_loss = sim_loss * params.get("sim_loss_coef", 1)
                    total_loss += sim_loss
                
                epoch_loss += total_loss

                with torch.no_grad():
                    wandb.log({
                        "total_loss": total_loss.cpu().detach().numpy(),
                        "lbox_loss": lbox.cpu().detach().numpy(),
                        "lcls_loss": lcls.cpu().detach().numpy(),
                        "tv_loss": tv_loss.cpu().detach().numpy(),
                        "green_penalty": green_penalty.cpu().detach().numpy(),
                        "similarity_loss": sim_loss.cpu().detach().numpy() if use_initial_image else 0,
                    })

                total_loss.backward()  # Compute loss gradient

                optimizer.step()  # Update delta
                optimizer.zero_grad()  # Clear gradients

                delta.data = torch.clamp(delta.data, min=0, max=1) 

        log_images = False
        # Calculate epoch summary statistics
        if epoch % save_freq == 0:
            log_images = True

        green_most_confident, red_still_there = attack_effectiveness(params, delta, device, model, val_dataloader, classes, log_images)        
        mean_epoch_loss = epoch_loss / (pgd_steps*len(train_dataloader))
        with torch.no_grad():
            wandb.log({"mean_epoch_loss": mean_epoch_loss,
                       "Most confidenct bbox is green (val split)": green_most_confident,
                       "Red bbox still present (val split)": red_still_there,
                       "epoch": epoch+1})     

        # Save patch
        if epoch%save_freq == 0:
            patch = delta.detach().cpu().numpy()
            patch = np.transpose(patch, (1, 2, 0))
            patch = wandb.Image(patch)
            wandb.log({"patch": patch})
            torch.save(delta.detach().cpu(), params["patch_name"] + str(epoch+1) + ".pt")


def attack_effectiveness(params, patch, device, model, dataloader, classes, log_images):
    """
    Calculate a percentage score for how effective the attack is over a dataset.
    Based upon the classification of the highest confidence bounding box.

    NOTE: this function assumes that each image has exactly one ground truth label
    TODO: if to be used with datasets other than mobile, this needs fixed
    """
    
    total_images_tested = 0
    images_successfully_attacked = 0
    still_had_red_bb = 0

    for (img, gt_label, _, _) in dataloader:
        # Move current image and label to device
        img = img.to(device)
        img = img.float() / 255.0
        gt_label = gt_label.to(device)

        # Skip this image if there are no labels, or no labels of the right colour
        if len(gt_label) != 1:
            continue
        gt_label = gt_label[0]
        if gt_label[1] != int(params.get("class_to_replace")):
            continue

        # Extract information from ground truth label
        img_size = params.get("img_size", 1280)
        x_center = gt_label[2].cpu().detach().numpy()*img_size
        y_center = gt_label[3].cpu().detach().numpy()*img_size
        bbox_width = gt_label[4].cpu().detach().numpy()*img_size
        bbox_height = gt_label[5].cpu().detach().numpy()*img_size

        # Resize patch to fit traffic light
        transformed_patch_size = int(bbox_width * params.get("patch_width_multiplier", 2))
        current_patch_resized = patch_resize(patch, (transformed_patch_size, transformed_patch_size))

        # Calculate patch coordinates
        patch_start_x = int(x_center - transformed_patch_size/2)
        patch_end_x = patch_start_x + transformed_patch_size
        patch_start_y = int(y_center + bbox_height/2)
        patch_end_y = patch_start_y + transformed_patch_size

        # If the patch is off the edge of the image
        if patch_end_x > img_size or patch_end_y > img_size  or patch_start_x < 0 or patch_start_y < 0:
            continue

        # Patch application
        for i in range(img.shape[0]):
            img[i, :, patch_start_y : patch_end_y, patch_start_x : patch_end_x] = current_patch_resized

        with torch.no_grad():
            out, _ = model(img)
            [out] = non_max_suppression(out.detach().cpu(), params["conf_thres"], params["iou_thres"])

        total_images_tested += 1
        
        
        if len(out) > 0:
            # If any red predictions made it past nms, the attack was unsuccessful
            if any(int(x) == int(params.get("class_to_replace")) for x in out[:, 5]):
                still_had_red_bb += 1
        
            # Find the prediction with the highest confidence
            _, index = torch.max(out[:, 4], 0)
            main_prediction = out[index]

            if int(main_prediction[5]) == int(params.get("target_class")):
                images_successfully_attacked += 1
    
    if log_images:
        # Log an image to see the progress on it
        bbox_adv_image = store_wandb_image_with_bboxes(img, [out], classes)
        wandb.log({"Adversarial image (from val split)": bbox_adv_image})

    green_most_confident = images_successfully_attacked / total_images_tested
    red_still_there = still_had_red_bb / total_images_tested
    return green_most_confident, red_still_there


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

                wandb.run.name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_universal_targeted_" + params["dataset"] + "_" + params["suffix"] + "_"+ str(params["img_size"])
                wandb.run.save()

                # Set random seeds
                random_seed = params.get("random_seed", 1)
                torch.manual_seed(random_seed)
                np.random.seed(random_seed)

                run_attack(params)

            except yaml.YAMLError as exc:
                print(exc)
