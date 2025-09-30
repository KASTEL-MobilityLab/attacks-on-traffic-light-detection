import kornia
from kornia.geometry import deg2rad, homography_warp, axis_angle_to_rotation_matrix

import torch
import torch.nn as nn


def patch_brightness(patch, factor):
    patch = kornia.color.rgb_to_hsv(patch)
    patch[:, 2] = patch[:, 2] * factor
    return torch.clamp(kornia.color.hsv_to_rgb(patch), 0.0001, 0.9999)


def patch_pad(patch, pad_width):
    padded_patch = torch.zeros(1, 3, (patch.shape[1] + 2 * pad_width), (patch.shape[1] + 2* pad_width)).cuda()
    padded_patch[:, :, pad_width:patch.shape[1]+pad_width, pad_width:patch.shape[2]+pad_width] = patch
    return padded_patch


def patch_rotate(patch, angle_x, angle_y, angle_z):
    angle = torch.ones(1, 3).cuda()
    angle[..., 0] = deg2rad(torch.ones(1).cuda()*angle_x)
    angle[..., 1] = deg2rad(torch.ones(1).cuda()*angle_y)
    angle[..., 2] = deg2rad(torch.ones(1).cuda()*angle_z)

    rotation_matrix = axis_angle_to_rotation_matrix(angle)
    target_patch_shape = (patch.shape[2], patch.shape[3])

    return homography_warp(patch, rotation_matrix, target_patch_shape)


def patch_resize(patch, size):
    if patch.dim() == 3:
        patch = patch.unsqueeze(0)
    return nn.Upsample(size=size, mode='bilinear')(patch)