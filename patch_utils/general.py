import io
import random

import matplotlib.patches as patches
import numpy as np
import PIL
import torch
import torchvision
import wandb
from matplotlib import pyplot as plt


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    with open(path, "r") as fp:
        names = fp.read().splitlines()
    classes = {i: names[i] for i in range(len(names))}
    return classes


def store_img_with_bboxes(img, detections, classes, title, colors):
    figure = plt.figure()
    fig, ax = plt.subplots(1)

    img_as_np = img.numpy()
    img_as_np = img_as_np[0]
    img_as_np = np.transpose(img_as_np, (1, 2, 0))
    ax.imshow((img_as_np * 255).astype(np.uint8))

    detections = detections[0]
    # unique_labels = detections[:, -1].cpu().unique()
    # n_cls_preds = len(unique_labels)
    # cmap = plt.get_cmap("tab20b")

    # set random colors for class
    # colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
    # bbox_colors = random.sample(colors, n_cls_preds)

    for x1, y1, x2, y2, conf, cls_pred in detections:
        # print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.2f}")

        box_w = x2 - x1
        box_h = y2 - y1
        # print(f"bbox_colors: {len(bbox_colors)}")
        # print(f"clas_pred: {int(cls_pred)}")
        color = colors[int(cls_pred)]
        # print(color)
        # Create a Rectangle patch
        bbox = patches.Rectangle(
            (x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none"
        )
        # Add the bbox to the plot
        ax.add_patch(bbox)
        # Add label
        plt.text(
            x1,
            y1,
            s=f"{classes[int(cls_pred)]}\n{conf.item():0.2f}",
            fontsize="xx-small",
            color="white",
            verticalalignment="top",
            bbox={"color": color, "pad": 0},
        )

    plt.title(title)
    # plt.show()
    return figure


def init_colors(n_cls):
    colors = []
    for i in range(n_cls):
        colors.append("#%06X" % random.randint(0, 0xFFFFFF))

    return colors


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to PIL image
    image = PIL.Image.open(buf)
    # convert to image tensor
    image = torchvision.transforms.ToTensor()(image)
    # Add the batch dimension
    image = torch.unsqueeze(image, 0)
    return image


def plot_images(img, detections, classes, colors, title="Clean Image"):
    figure = store_img_with_bboxes(
        img.detach().cpu(), detections, classes, title, colors
    )
    image = plot_to_image(figure)
    img_grid = torchvision.utils.make_grid(image[:1].detach().cpu(), normalize=True)
    return img_grid


def store_wandb_image_with_bboxes(img, detections, classes):
    # load raw input photo
    raw_image = img.detach().cpu().numpy()[0]
    raw_image = np.transpose(raw_image, (1, 2, 0))

    all_boxes = []
    detections = detections[0]
    # plot each bounding box for this image
    for x1, y1, x2, y2, conf, cls_pred in detections:
        # print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.2f}")
        # get coordinates and labels
        box_data = {
            "position": {
                "minX": x1.item(),
                "maxX": x2.item(),
                "minY": y1.item(),
                "maxY": y2.item(),
            },
            "class_id": int(cls_pred.item()),
            # optionally caption each box with its class and score
            "box_caption": "%s (%.3f)" % (classes[int(cls_pred)], conf.item()),
            "domain": "pixel",
            "scores": {"score": conf.item()},
        }
        all_boxes.append(box_data)
    # log to wandb: raw image, predictions, and dictionary of class labels for each class id

    bbox_image = wandb.Image(
        raw_image,
        boxes={"predictions": {"box_data": all_boxes, "class_labels": classes}},
    )
    return bbox_image
