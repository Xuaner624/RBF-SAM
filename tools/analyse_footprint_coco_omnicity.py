import json
import sys

sys.path.append("./")
import os
from pycocotools.coco import COCO
import numpy as np
import pickle as pkl
from tabulate import tabulate
import cv2
from pycocotools import mask as maskUtils
from multiprocessing.pool import Pool
from tqdm import tqdm
import warnings


def roof_offset2foot_print(mask, offset):
    seg2 = mask
    translation_matrix = np.float32([[1, 0, -offset[0]], [0, 1, -offset[1]]])
    seg_ = cv2.warpAffine(seg2, translation_matrix, (512, 512))  # building layer
    return seg_


def mask_iou(mask1, mask2):
    mask1 = np.asarray(mask1).astype(bool)
    mask2 = np.asarray(mask2).astype(bool)
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    iou = intersection / union if union != 0 else 0.0
    return iou


def decode_mask(mask, size=512):
    if not isinstance(mask, list):
        mask = [mask]
    elif not isinstance(mask[0], list):
        mask = [mask]

    if not isinstance(mask[0], dict):

        mask = maskUtils.merge(maskUtils.frPyObjects(mask, size, size))
        mask = maskUtils.decode(mask)
    else:
        mask = maskUtils.merge(mask)
        mask = maskUtils.decode(mask)
    return mask


def myf1score(pred, ann):
    TP = np.sum(pred & ann)
    FP = np.sum(pred & ~ann)
    FN = np.sum(~pred & ann)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    return (precision, recall, f1)


def compare_f1score(coco: COCO,
                    pred_list: list,
                    start_len=None,
                    end_len=None,
                    ):

    def _crop_mask(mask, bbox):
        x1, y1, x2, y2 = [int(a) for a in bbox]
        return mask[y1:y2 + y1, x1:x1 + x2]

    def _detect_in_range(ann_len, start_len=start_len, end_len=end_len):
        assert start_len is not None or end_len is not None
        if start_len is None:
            return ann_len < end_len
        elif end_len is None:
            return start_len <= ann_len
        else:
            return start_len <= ann_len and ann_len < end_len

    precision = []
    recall = []
    fscore = []
    ious = []

    pred_list = pred_list if isinstance(pred_list, list) else pred_list['annotations']
    img_ids = list(set([pred['image_id'] for pred in pred_list]))
    img_ids2pred = {img_id: [] for img_id in img_ids}
    for pred in pred_list:
        len_offset = (pred['offset'][0] ** 2 + pred['offset'][1] ** 2) ** 0.5
        if _detect_in_range(len_offset):
            img_ids2pred[pred['image_id']].append(pred)

    for img_id in tqdm(img_ids):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        preds = img_ids2pred[img_id]
        if len(preds) == 0:
            continue
        ann_foot = [decode_mask(ann['footprint_mask']) for ann in anns]
        # ann_foot = [
        #     decode_mask(ann['footprint_mask'])
        #     for ann in anns
        #     if ann.get('iscrowd', 0) == 0
        # ]
        prd_foot = [decode_mask(pred['footprint_mask']) for pred in preds]
        af = sum(ann_foot) > 0
        pf = sum(prd_foot) > 0

        precision_recall_fscore = myf1score(pf, af)
        if not np.isnan(precision_recall_fscore[2]):
            precision.append(precision_recall_fscore[0])
            recall.append(precision_recall_fscore[1])
            fscore.append(precision_recall_fscore[2])
            ious.append(mask_iou(pf, af))

    print('done')
    return [start_len, end_len, np.mean(precision), np.mean(recall), np.mean(fscore), np.mean(ious)]


def compare_f1score_roof2foot(coco: COCO,
                              pred_list: list,
                              start_len=None,
                              end_len=None,
                              ):
    def _crop_mask(mask, bbox):
        x1, y1, x2, y2 = [int(a) for a in bbox]
        return mask[y1:y2 + y1, x1:x1 + x2]

    def _detect_in_range(ann_len, start_len=start_len, end_len=end_len):
        assert start_len is not None or end_len is not None
        if start_len is None:
            return ann_len < end_len
        elif end_len is None:
            return start_len <= ann_len
        else:
            return start_len <= ann_len and ann_len < end_len

    precision = []
    recall = []
    fscore = []
    ious = []
    # mean = []
    pred_list = pred_list if isinstance(pred_list, list) else pred_list['annotations']
    # pred_list = [ann for anns in pred_list for ann in anns]
    img_ids = list(set([pred['image_id'] for pred in pred_list]))
    img_ids2pred = {img_id: [] for img_id in img_ids}
    for pred in pred_list:
        len_offset = (pred['offset'][0] ** 2 + pred['offset'][1] ** 2) ** 0.5
        if _detect_in_range(len_offset):
            img_ids2pred[pred['image_id']].append(pred)
    for img_id in tqdm(img_ids):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        preds = img_ids2pred[img_id]
        if len(preds) == 0:
            continue

        ann_foot = [
            decode_mask(ann['footprint_mask'])
            for ann in anns
            if ann.get('iscrowd', 0) == 0
        ]
        prd_foot = [roof_offset2foot_print(decode_mask(pred['roof_mask']), np.array(pred['offset']) / SCALE) for pred in
                    preds]
        af = sum(ann_foot) > 0
        pf = sum(prd_foot) > 0

        precision_recall_fscore = myf1score(pf, af)
        if not np.isnan(precision_recall_fscore[2]):
            precision.append(precision_recall_fscore[0])
            recall.append(precision_recall_fscore[1])
            fscore.append(precision_recall_fscore[2])
            ious.append(mask_iou(pf, af))

    print('done')
    return [start_len, end_len, np.mean(precision), np.mean(recall), np.mean(fscore), np.mean(ious)]


ann = COCO('/root/autodl-tmp/data/OmniCityView3WithOffset/coco/OmniCityView3WithOffset_test_roof.json')

def measure_detail(pred_path, anns=ann):
    preds = json.load(open(pred_path))
    range_len = []
    txt = os.path.join('/root/autodl-tmp/RBF-SAM-main/results/rbf_sam_omnicity/',
                       os.path.basename(pred_path).replace('.json', '.txt'))
    headers = ['start_len', 'end_len', 'precision', 'recall', 'f1score', 'iou']

    try:
        _, _, p, r, f1, iou = compare_f1score(anns, preds, 0, None)
        range_len.append(['-', '-', '-', '-', '-'])
        range_len.append(['footprint polygon', None])
        range_len.append([None, None, p, r, f1, iou])
    except:
        print('footprint polygon not found !!!!')

    _, _, p, r, f1, iou = compare_f1score_roof2foot(anns, preds, 0, None)
    range_len.append(['-', '-', '-', '-', '-'])
    range_len.append(['roof mask to footprint', None])
    range_len.append([None, None, p, r, f1, iou])
    # ---------------- 保存和打印结果 ----------------
    table = tabulate(range_len, headers, tablefmt="pipe")
    f = open(txt, 'a')
    print(txt, file=f)
    print(table, file=f)
    print(table)
    f.close()


def fun_wapper(args):
    fun, p, anns = args
    return fun(p, anns)


# python tools/analyse_footprint_coco_omnicity.py
if __name__ == '__main__':
    SCALE = 2.0
    pred_path = [
        '/root/autodl-tmp/RBF-SAM-main/results/rbf_sam_omnicity/rbf_sam_omnicity.json',
    ]
    COCO_path = [
        '/root/autodl-tmp/data/OmniCityView3WithOffset/coco/OmniCityView3WithOffset_test_roof.json',
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not os.path.exists('results/rbf_sam_omnicity'):
            os.mkdir('results/rbf_sam_omnicity')
        print(pred_path)
        for n, m in zip(pred_path, COCO_path):
            measure_detail(n, anns=COCO(m))
