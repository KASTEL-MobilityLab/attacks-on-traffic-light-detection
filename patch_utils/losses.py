import torch
import torch.nn as nn
import torchvision.transforms as T

# from utils.general import bbox_iou, xyxy2xywh
from ultralytics.utils.metrics import bbox_iou
# from utils.loss import ComputeLoss, smooth_BCE
# from utils.torch_utils import is_parallel


def total_variation_loss(patch: torch.Tensor) -> torch.Tensor:
    """
    Given a patch, calculate the total pixel variation.
    Higher value means 'sharper' transitions between pixels,
    lower score means a more smooth patch.
    :param patch: 4D tensor of shape (batch_size, channels, height, width)
    :returns tv_loss: Total variation loss as a single scalar tensor.
    """

    # Calculate pixel differences in height and width directions
    diff_h = patch[:, :, 1:, :] - patch[:, :, :-1, :]
    diff_w = patch[:, :, :, 1:] - patch[:, :, :, :-1]

    # Compute the total variation loss
    tv_h = torch.pow(diff_h, 2).sum()
    tv_w = torch.pow(diff_w, 2).sum()

    # Normalize the loss by the number of elements in each dimension (excluding batch and channels)
    num_elements = patch.size(2) * patch.size(3)
    tv_loss = (tv_h + tv_w) / num_elements

    return tv_loss


def similarity_loss(
        current_patch: torch.Tensor,
        initial_patch: torch.Tensor):
    
    """
    Used when training a patch to look similar to an image.
    """

    return nn.MSELoss()(current_patch, initial_patch)


def green_channel_penalty(
        patch: torch.Tensor, 
        threshold=0.5, 
        blur_kernel_size=5, 
        blur_sigma=1.0
    ):  
    """
    Given a patch, calculate a score representing the number of
    clusters of green pixels in the image.
    """

    green_channel = patch[1, :, :].unsqueeze(0).unsqueeze(0)
    
    # Apply blur to ignore single green pixels
    gaussian_blur = T.GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)
    blurred_green_channel = gaussian_blur(green_channel)
    blurred_green_channel = blurred_green_channel.squeeze()

    green_mask = blurred_green_channel > threshold
    penalty = green_mask.sum().float()

    return penalty


def yolo_loss(model, p, targets, device):
    """
    Modified YOLOv7 loss function.
    :returns lbox: Bounding box position loss component.
    :returns lcls: Class confidence loss component.
    """
    compute_loss = ComputeLoss(model)
    h = model.hyp
    cp, cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls, tbox, indices, anchors = compute_loss.build_targets(p, targets)  # targets

    det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
    balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])
    # Losses
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            lbox += (1.0 - iou).mean()  # iou loss

            # Objectness
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

            # Classification
            t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
            t[range(n), tcls[i]] = cp
            lcls += BCEcls(ps[:, 5:], t)  # BCE

        obji = BCEobj(pi[..., 4], tobj)
        lobj += obji * balance[i]  # obj loss

    return lbox, lobj, lcls 

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from ultralytics.utils.metrics import bbox_iou, probiou
from ultralytics.utils.tal import bbox2dist


from ultralytics.utils.loss import BboxLoss

class v8AttackLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        cls_loss = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
#         if fg_mask.sum():
#             target_bboxes /= stride_tensor
#             box_loss, dfl_loss = self.bbox_loss(
#                 pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
#             )

        return cls_loss # box_loss, cls_loss, dfl_loss  # loss(box, cls, dfl)

