import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist


def calculate_hd95(pred, target, voxel_spacing=1):
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    # Extract foreground coordinates
    pred_coords = np.transpose(np.nonzero(pred_np))
    target_coords = np.transpose(np.nonzero(target_np))

    # If one mask is empty → HD95 is undefined (return inf)
    if len(pred_coords) == 0 or len(target_coords) == 0:
        return np.inf

    # Apply voxel spacing
    pred_coords_scaled = pred_coords * voxel_spacing
    target_coords_scaled = target_coords * voxel_spacing

    # Compute pairwise distances
    dists = cdist(pred_coords_scaled, target_coords_scaled)

    # Directed distances
    pred_to_target = np.min(dists, axis=1)
    target_to_pred = np.min(dists, axis=0)

    # Symmetric distances
    all_dists = np.concatenate([pred_to_target, target_to_pred])

    # 95th percentile
    hd95 = np.percentile(all_dists, 95)

    return hd95


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    # we need to use sigmoid because the output of Unet is logit.
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def dice_coef2(output, target):
    "This metric is for validation purpose"
    smooth = 1e-5

    output = output.view(-1)
    output = (output > 0.5).float().cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
