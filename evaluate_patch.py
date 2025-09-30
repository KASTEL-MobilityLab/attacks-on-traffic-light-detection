import sys
sys.path.append("yolov7/")

import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.experimental import attempt_load
from utils.general import coco80_to_coco91_class, check_img_size, box_iou, non_max_suppression, \
      scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel

from patch_utils.datasets import load_dataset
from patch_utils.general import load_classes

def resize_patch(patch, size):
    return nn.Upsample(size=size, mode='bilinear')(torch.unsqueeze(patch, 0))

def test(test_dataloader,
         classes,
         trained_patch_path,
         bbox_width_multiplier,
         patch_start_h,
         patch_start_w,
         weights=None,
         batch_size=32,
         imgsz=1280,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         trace=False,
         is_coco=False,
         v5_metric=False,
         bin_images=False):

    # Initial setup
    # Initialize/load model and set device
    training = model is not None
    
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

        if trace:
            model = TracedModel(model, device, imgsz)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    nc = len(classes)  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel() # number of iou thresholds

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)

    if v5_metric:
        print("Testing with YOLOv5 AP metric...")

    # Patch loading
    delta = torch.load(trained_patch_path).to(device)
    
    seen = 0

    # Initialise confusion matrices
    if bin_images:
        size_bins = {
            'small': (0, 25*25),
            'medium': (25*25, 50*50),
            'large': (50*50, float('inf'))
        }
        confusion_matrices = {key: ConfusionMatrix(nc=nc) for key in size_bins.keys()}
    else:
        confusion_matrix = ConfusionMatrix(nc=nc)
    
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    

    # Perform evaluation over dataset
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(test_dataloader, desc=s)):
        
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        # Check that there are bboxes labelled in this image
        targets = targets.to(device)
        gt_label = targets.clone()
        if len(gt_label) == 0:
            continue

        # Place patches below traffic lights
        # gt_label in format (class_id, x_centre,  y_centre,  width,  height)
        for i, gtx in enumerate(gt_label):      
            x_center = gtx[2].cpu().detach().numpy()*imgsz
            y_center = gtx[3].cpu().detach().numpy()*imgsz
            bbox_width = gtx[4].cpu().detach().numpy()*imgsz
            bbox_height = gtx[5].cpu().detach().numpy()*imgsz
            
            patch_size = bbox_width * bbox_width_multiplier
            patch_start_x = max(0, int(x_center - patch_size / 2))
            patch_end_x = min(img.shape[3], int(patch_start_x + patch_size))
            patch_start_y = max(0, int(y_center + bbox_height / 2))
            patch_end_y = min(img.shape[2], int(patch_start_y + patch_size))
            
            resized_patch = resize_patch(delta, (patch_end_y - patch_start_y, patch_end_x - patch_start_x))

            img[i, :, patch_start_y:patch_end_y, patch_start_x:patch_end_x] = resized_patch

        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            
            labels = targets[targets[:, 0] == si, 1:]
            num_labels = len(labels)
            target_class = labels[:, 0].tolist() if num_labels else []  # target class
            path = Path(paths[si])
            seen += 1

            # YOLO failed to detect the traffic light
            if len(pred) == 0:
                if num_labels:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), target_class))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Optional additional logging
            # Append to text file (disabled by default)
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots (disabled by default)
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary (disabled by default)
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if num_labels:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

                if bin_images:
                    # Calculate areas of target bboxes
                    pred_areas = (predn[:, 2] - predn[:, 0]) * (predn[:, 3] - predn[:, 1])
                    gt_areas = (tbox[:, 2] - tbox[:, 0]) * (tbox[:, 3] - tbox[:, 1])

                    if plots:
                        for key, (min_area, max_area) in size_bins.items():
                            pred_size_mask = (pred_areas >= min_area) & (pred_areas < max_area)
                            gt_size_mask = (gt_areas >= min_area) & (gt_areas < max_area)
                            confusion_matrices[key].process_batch(
                                predn[pred_size_mask], torch.cat((labels[gt_size_mask, 0:1], tbox[gt_size_mask]), 1)
                        )
                else:
                    if plots:
                       confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == num_labels:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, target_class)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), target_class))

        # Plot images
        if plots and batch_i % 100 == 0:

            # Save ground truth image
            f = save_dir / f'test_batch{batch_i}_labels.jpg'
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            
            # Save image with predictions
            f = save_dir / f'test_batch{batch_i}_pred.jpg'
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()
 
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        if bin_images:
            for key, cm in confusion_matrices.items():
                cm.plot(save_dir=save_dir, filename=str(key), names=list(names.values()))
        else:
            confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in test_dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(test_dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument("--test_imgs_path", type=str, default="data")
    parser.add_argument("--n_cpu", type=int, default=8)
    parser.add_argument("--logs_path", type=str, default="./logs")
    parser.add_argument("--logs_name", type=str, default="pgd_patch")
    parser.add_argument("--configs_path", type=str, default="./configs")
    parser.add_argument("--bbox_width_multiplier", type=int, default=2)
    parser.add_argument("--patch_start_h", type=int, default=0, help="Start height for the patch position.")
    parser.add_argument("--patch_start_w", type=int, default=0, help="Start width for the patch position.")
    parser.add_argument("--trained_patch_path", type=str, default="./")
    parser.add_argument("--classes_path", type=str, default="configs/dtld/classes.names")
    parser.add_argument("--bin_images", default=False, help='create confusion matrices for images of different sizes')


    opt = parser.parse_args()
    #check_requirements()

    test_dataloader = load_dataset(opt.test_imgs_path, opt.img_size, opt.batch_size, opt.n_cpu)
    classes = load_classes(opt.classes_path)

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(test_dataloader,
             classes,
             opt.trained_patch_path,
             opt.bbox_width_multiplier,
             opt.patch_start_h,
             opt.patch_start_w,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             trace=not opt.no_trace,
             v5_metric=opt.v5_metric,
             bin_images=opt.bin_images
             )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, v5_metric=opt.v5_metric)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.65 --weights yolov7.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, v5_metric=opt.v5_metric)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
