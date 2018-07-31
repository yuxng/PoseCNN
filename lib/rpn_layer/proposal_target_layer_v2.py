# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick, Sean Bell and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division

import numpy as np
import numpy.random as npr
from fcn.config import cfg
from utils.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps

def proposal_target_layer_v2(rpn_rois, rpn_scores, gt_boxes, poses, _num_classes):
  """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  """

  # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN --> (batch_ids, x1, y1, x2, y2)
  # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
  # gt_boxes (batch_ids, x1, y1, x2, y2, cls)
  all_rois = rpn_rois
  all_scores = rpn_scores

  # Sample rois with classification labels and bounding box regression
  # targets
  labels, rois, roi_scores, bbox_targets, bbox_inside_weights, poses_target, poses_weight = _sample_rois(
    all_rois, all_scores, gt_boxes, poses, _num_classes)

  rois = rois.reshape(-1, 5)
  roi_scores = roi_scores.reshape(-1)
  labels = labels.reshape(-1, 1)
  bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
  bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
  bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

  # convert labels
  num = labels.shape[0]
  label_blob = np.zeros((num, 2), dtype=np.float32)
  for i in xrange(num):
      if labels[i] == 0:
          label_blob[i, 0] = 1.0
      else:
          label_blob[i, 1] = 1.0

  return rois, roi_scores, label_blob, bbox_targets, bbox_inside_weights, bbox_outside_weights, poses_target, poses_weight


def _get_bbox_regression_labels(bbox_target_data, num_classes):
  """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tw, th)

  This function expands those targets into the 4-of-4*K representation used
  by the network (i.e. only one class has non-zero targets).

  Returns:
      bbox_target (ndarray): N x 4K blob of regression targets
      bbox_inside_weights (ndarray): N x 4K blob of loss weights
  """

  clss = bbox_target_data[:, 0]
  bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
  bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
  inds = np.where(clss > 0)[0]
  for ind in inds:
    cls = clss[ind]
    start = int(4 * cls)
    end = start + 4
    bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
    bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
  return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
  """Compute bounding-box regression targets for an image."""

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 4

  targets = bbox_transform(ex_rois, gt_rois)
  if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    # Optionally normalize targets by a precomputed mean and stdev
    targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
               / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
  return np.hstack(
    (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


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


def _sample_rois(all_rois, all_scores, gt_boxes, poses, num_classes):
  """Generate a random sample of RoIs comprising foreground and background
  examples.
  """
  # all_rois (batch_ids, x1, y1, x2, y2)
  # gt_boxes (batch_ids, x1, y1, x2, y2, cls)
  # overlaps: (rois x gt_boxes)
  overlaps = bbox_overlaps(
    np.ascontiguousarray(all_rois[:, :5], dtype=np.float),
    np.ascontiguousarray(gt_boxes[:, :5], dtype=np.float))

  gt_assignment = overlaps.argmax(axis=1)
  max_overlaps = overlaps.max(axis=1)
  labels = gt_boxes[gt_assignment, 5]
  quaternions = poses[gt_assignment, 6:10]

  # Select foreground RoIs as those with >= FG_THRESH overlap
  fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
  bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH)[0]

  print '{:d} rois, {:d} fg, {:d} bg'.format(all_rois.shape[0], len(fg_inds), len(bg_inds))
  # print all_rois

  # The indices that we're selecting (both fg and bg)
  keep_inds = np.append(fg_inds, bg_inds)
  # Select sampled values from various arrays:
  labels = labels[keep_inds]
  # Clamp labels for the background RoIs to 0
  labels[int(len(fg_inds)):] = 0
  rois = all_rois[keep_inds]
  roi_scores = all_scores[keep_inds]

  # pose regression targets and weights
  poses_target, poses_weight = _compute_pose_targets(quaternions[keep_inds], labels, num_classes)

  bbox_target_data = _compute_targets(
    rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], 1:5], labels)

  bbox_targets, bbox_inside_weights = \
    _get_bbox_regression_labels(bbox_target_data, num_classes)

  return labels, rois, roi_scores, bbox_targets, bbox_inside_weights, poses_target, poses_weight
