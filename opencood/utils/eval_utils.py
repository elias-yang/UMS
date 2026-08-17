import os

import numpy as np
import torch

from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh,
                    left_range=-float('inf'), right_range=float('inf')):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    right_range : float
        The evaluarion range right bound
    left_range : float
        The evaluation range left bound
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        det_polygon_list_origin = list(common_utils.convert_format(det_boxes))
        gt_polygon_list_origin = list(common_utils.convert_format(gt_boxes))
        det_polygon_list = []
        gt_polygon_list = []
        det_score_new = []
        # remove the bbx out of range
        for i in range(len(det_polygon_list_origin)):
            det_polygon = det_polygon_list_origin[i]
            distance = np.sqrt(det_polygon.centroid.x**2 +
                               det_polygon.centroid.y**2)
            if left_range < distance < right_range:
                det_polygon_list.append(det_polygon)
                det_score_new.append(det_score[i])

        for i in range(len(gt_polygon_list_origin)):
            gt_polygon = gt_polygon_list_origin[i]
            distance = np.sqrt(gt_polygon.centroid.x**2 +
                               gt_polygon.centroid.y**2)
            if left_range < distance < right_range:
                gt_polygon_list.append(gt_polygon)

        gt = len(gt_polygon_list)
        det_score_new = np.array(det_score_new)
        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score_new)
        sorted_scores = det_score_new[score_order_descend].tolist()

        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)
    else:
        gt = gt_boxes.shape[0]
    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt
    if 'score' in result_stat[iou_thresh]:
        result_stat[iou_thresh]['score'] += sorted_scores \
            if det_boxes is not None else []


def calculate_metrics(result_stat, iou):
    """
    Calculate AP and final precision/recall at one IoU threshold.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
    iou : float
    """
    iou_stat = result_stat[iou]
    fp = np.asarray(iou_stat['fp'], dtype=np.float64)
    tp = np.asarray(iou_stat['tp'], dtype=np.float64)
    assert len(fp) == len(tp)
    gt_total = int(iou_stat['gt'])

    scores = np.asarray(iou_stat.get('score', []), dtype=np.float64)
    if len(scores) == len(tp):
        order = np.argsort(-scores, kind='stable')
        fp = fp[order]
        tp = tp[order]

    tp_total = int(tp.sum())
    fp_total = int(fp.sum())
    precision = tp_total / (tp_total + fp_total) \
        if tp_total + fp_total > 0 else 0.0
    recall = tp_total / gt_total if gt_total > 0 else 0.0

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    if gt_total > 0:
        recall_curve = cumulative_tp / gt_total
    else:
        recall_curve = np.zeros_like(cumulative_tp)
    precision_curve = np.divide(
        cumulative_tp,
        cumulative_tp + cumulative_fp,
        out=np.zeros_like(cumulative_tp),
        where=(cumulative_tp + cumulative_fp) > 0)
    ap, mrec, mprec = voc_ap(recall_curve.tolist(),
                              precision_curve.tolist())

    return {
        'ap': float(ap),
        'precision': float(precision),
        'recall': float(recall),
        'tp': tp_total,
        'fp': fp_total,
        'gt': gt_total,
        'mrec': mrec,
        'mpre': mprec,
    }


def calculate_ap(result_stat, iou):
    """Return AP and the interpolated recall/precision curves."""
    metrics = calculate_metrics(result_stat, iou)
    return metrics['ap'], metrics['mrec'], metrics['mpre']


def eval_final_results(result_stat, save_path, range="", epoch=None):
    """Print and save AP, precision, and recall for all IoU thresholds."""
    metrics = {
        iou: calculate_metrics(result_stat, iou)
        for iou in sorted(result_stat)
    }
    dump_dict = {}
    for iou, values in metrics.items():
        suffix = int(round(iou * 100))
        for key in ('ap', 'precision', 'recall', 'tp', 'fp', 'gt',
                    'mpre', 'mrec'):
            dump_dict[f'{key}_{suffix}'] = values[key]

    if epoch is None:
        file_name = 'eval.yaml' if range == "" else range + '_eval.yaml'
    else:
        prefix = 'eval' if range == "" else range + '_eval'
        file_name = f'{prefix}_epoch_{epoch:03d}.yaml'
    if save_path is not None:
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, file_name))

    title = f'Test metrics - epoch {epoch}' if epoch is not None \
        else 'Evaluation metrics'
    if range:
        title += f' - range {range}'
    print('=' * 62)
    print(title)
    print(' IoU |    AP    | Precision |  Recall  |  TP  |  FP  |  GT')
    print('-' * 62)
    for iou, values in metrics.items():
        print(f' {iou:3.1f} | {values["ap"]:8.4f} | '
              f'{values["precision"]:9.4f} | {values["recall"]:8.4f} | '
              f'{values["tp"]:4d} | {values["fp"]:4d} | '
              f'{values["gt"]:4d}')
    print('=' * 62)
    return metrics
