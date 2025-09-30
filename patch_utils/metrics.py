import numpy as np
import torch
from utils.general import box_iou, scale_coords, xywh2xyxy
from utils.metrics import compute_ap


def ap_per_class(tp, conf, pred_cls, target_cls, v5_metric=False, plot=True):
    """Compute the average precision, given the recall and precision curves.
    Source: https://github.com/WongKinYiu/yolov7/
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(
                -px, -conf[i], recall[:, 0], left=0
            )  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(
                    recall[:, j], precision[:, j], v5_metric=v5_metric
                )
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype("int32")


def calculate_stats_for_batch(img, out, gt_label, iouv, niou, shapes, device="cpu"):
    stats = []
    for si, pred in enumerate(out):
        labels = gt_label[gt_label[:, 0] == si, 1:]
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class

        if len(pred) == 0:
            if nl:
                stats.append(
                    (
                        torch.zeros(0, niou, dtype=torch.bool),
                        torch.Tensor(),
                        torch.Tensor(),
                        tcls,
                    )
                )
            continue

        # Predictions
        predn = pred.clone()
        scale_coords(
            img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1]
        )  # native-space pred

        # Assign all predictions as incorrect
        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
        if nl:
            detected = []  # target indices
            tcls_tensor = labels[:, 0]

            # target boxes
            tbox = xywh2xyxy(labels[:, 1:5])
            scale_coords(
                img[si].shape[1:], tbox, shapes[si][0], shapes[si][1]
            )  # native-space labels

            # Per target class
            for cls in torch.unique(tcls_tensor):
                ti = (
                    (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                )  # prediction indices
                pi = (
                    (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)
                )  # target indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = box_iou(predn[pi, :4], tbox[ti]).max(
                        1
                    )  # best ious, indices

                    # Append detections
                    detected_set = set()
                    for j in (ious > iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]  # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if (
                                len(detected) == nl
                            ):  # all targets already located in image
                                break

        # Append statistics (correct, conf, pcls, tcls)
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        return stats
