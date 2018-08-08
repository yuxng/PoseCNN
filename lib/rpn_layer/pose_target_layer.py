# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from fcn.config import cfg
from utils.bbox_transform import bbox_transform_inv, clip_boxes
from utils.nms_wrapper import nms
from utils.cython_bbox import bbox_overlaps

# rpn_rois, gt_boxes: (batch_ids, x1, y1, x2, y2, cls)
def pose_target_layer(rois, bbox_pred, im_info, gt_boxes, poses, num_classes):
  
    # process boxes
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (num_classes))
        means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (num_classes))
        bbox_pred *= stds
        bbox_pred += means

    boxes = rois[:, 1:5].copy()
    pred_boxes = bbox_transform_inv(boxes, bbox_pred)
    pred_boxes = clip_boxes(pred_boxes, im_info[:2])

    # assign boxes
    for i in xrange(rois.shape[0]):
        cls = int(rois[i, 5])
        rois[i, 1:5] = pred_boxes[i, cls*4:cls*4+4]

    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(rois[:, :5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :5], dtype=np.float))

    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 5]
    quaternions = poses[gt_assignment, 6:10]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH_POSE)[0]
    labels[bg_inds] = 0

    bg_inds = np.where(rois[:, -1] != labels)[0]
    labels[bg_inds] = 0
    
    # pose regression targets and weights
    poses_target, poses_weight = _compute_pose_targets(quaternions, labels, num_classes)

    ind = [0, 5, 1, 2, 3, 4]
    rois_target = rois[:, ind]

    return rois_target, poses_target, poses_weight


def _compute_pose_targets(quaternions, labels, num_classes):
    """Compute pose regression targets for an image."""

    num = quaternions.shape[0]
    poses_target = np.zeros((num, 4 * num_classes), dtype=np.float32)
    poses_weight = np.zeros((num, 4 * num_classes), dtype=np.float32)

    for i in xrange(num):
        cls = labels[i]
        if cls > 0:
            start = int(4 * cls)
            end = start + 4
            poses_target[i, start:end] = quaternions[i, :]
            poses_weight[i, start:end] = 1.0

    return poses_target, poses_weight
