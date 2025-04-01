import math

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def iou(x, y, axis):
    iou_ = (x & y).sum(axis) / (x | y).sum(axis)
    iou_[torch.isnan(iou_)] = 1.0
    return iou_


def batched_distance(x, y):
    per_class_iou = iou(x[:, :, None], y[:, None, :], axis=-2)
    # Exclude background
    return 1 - per_class_iou[..., 1:].mean(-1)


def batched_ged(samples_dist_0, samples_dist_1, num_classes):
    samples_dist_0 = samples_dist_0.reshape(*samples_dist_0.shape[:2], -1)
    samples_dist_1 = samples_dist_1.reshape(*samples_dist_1.shape[:2], -1)

    samples_dist_0 = F.one_hot(samples_dist_0, num_classes)
    samples_dist_1 = F.one_hot(samples_dist_1, num_classes)

    cross = batched_distance(samples_dist_0, samples_dist_1).mean(dim=(1, 2))
    diversity_0 = batched_distance(samples_dist_0, samples_dist_0).mean(dim=(1, 2))
    diversity_1 = batched_distance(samples_dist_1, samples_dist_1).mean(dim=(1, 2))
    return 2 * cross - diversity_0 - diversity_1, diversity_0, diversity_1


def batched_hungarian_iou(samples_dist_0, samples_dist_1, num_classes):
    samples_dist_0 = samples_dist_0.reshape(*samples_dist_0.shape[:2], -1)
    samples_dist_1 = samples_dist_1.reshape(*samples_dist_1.shape[:2], -1)

    samples_dist_0 = F.one_hot(samples_dist_0, num_classes)
    samples_dist_1 = F.one_hot(samples_dist_1, num_classes)

    n_samples_0 = samples_dist_0.size(1)
    n_samples_1 = samples_dist_1.size(1)

    lcm = math.lcm(n_samples_0, n_samples_1)

    samples_dist_0 = torch.repeat_interleave(samples_dist_0, lcm // n_samples_0, dim=1)
    samples_dist_1 = torch.repeat_interleave(samples_dist_1, lcm // n_samples_1, dim=1)

    cost_matrices = batched_distance(samples_dist_0, samples_dist_1)
    h_scores = []

    for cost_matrix in cost_matrices:
        iou_matrix = 1 - cost_matrix
        h_scores.append(iou_matrix[linear_sum_assignment(cost_matrix.cpu())].mean())

    return torch.stack(h_scores)
