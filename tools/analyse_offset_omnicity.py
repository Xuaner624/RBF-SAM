import json
import os
from pycocotools.coco import COCO
import numpy as np
import pickle as pkl
from tabulate import tabulate


def compare_ordered_json(ann_list: COCO,
                         pred_list: list,
                         check=False,
                         ):
    mean_dis = []
    mean_len = []
    mean_ang = []
    for pred in pred_list:
        ann = ann_list.loadAnns(pred['id'])[0]
        ann_offset = np.array(ann["offset"])
        pred_offset = np.array(pred['offset'][:2]) / 2
        distance = np.sqrt(np.sum((ann_offset - pred_offset) ** 2))
        ann_len = np.sqrt(np.sum(ann_offset ** 2))
        pred_len = np.sqrt(np.sum(pred_offset ** 2))
        a = sum(ann_offset * pred_offset) / (np.linalg.norm(ann_offset) * np.linalg.norm(pred_offset))
        if check:
            if sum(ann['building_bbox']) != sum(pred['building_bbox']):
                # print(pred["file_name"])
                continue
        # print('yes')
        if not np.isnan(a):
            mean_ang.append(np.arccos(a))
            # mean_dis.append(distance)
            # mean_len.append(abs(ann_len-pred_len))
        else:
            mean_ang.append(0)
            # mean_dis.append(0)
            # mean_len.append(0)
        mean_dis.append(distance)
        mean_len.append(abs(ann_len - pred_len))
        # mean_len.append(ann_len-pred_len)

    return mean_dis, mean_len, mean_ang


def compare_ordered_json_AL(coco: COCO,
                            pred_list: list,
                            start_len=None,
                            end_len=None,
                            ):

    def _detect_in_range(ann_len, start_len=start_len, end_len=end_len):
        assert start_len is not None or end_len is not None
        if start_len is None:
            return ann_len < end_len
        elif end_len is None:
            return start_len <= ann_len
        else:
            return start_len <= ann_len and ann_len < end_len

    mean_dis = []
    mean_len = []
    mean_ang = []
    for pred in pred_list:
        ann = coco.loadAnns(pred['id'])[0]
        if ann.get('iscrowd', 0) == 1:
            continue
        ann_offset = np.array(ann["offset"])
        pred_offset = np.array(pred['offset'][:2]) / 2
        distance = np.sqrt(np.sum((ann_offset - pred_offset) ** 2))
        ann_len = np.sqrt(np.sum(ann_offset ** 2))
        pred_len = np.sqrt(np.sum(pred_offset ** 2))
        a = sum(ann_offset * pred_offset) / (np.linalg.norm(ann_offset) * np.linalg.norm(pred_offset))
        if not _detect_in_range(ann_len):
            continue

        if not np.isnan(a):
            mean_ang.append(np.arccos(a))
            # mean_dis.append(distance)
            # mean_len.append(abs(ann_len-pred_len))
        else:
            mean_ang.append(0)
            # mean_dis.append(0)
            # mean_len.append(0)
        mean_dis.append(distance)
        mean_len.append(abs(ann_len - pred_len))
    return [start_len, end_len, np.mean(mean_dis), np.mean(mean_len), np.mean(mean_ang)]


def measure_detail(anns, preds, txt):
    headers = ['start_len', 'end_len', 'mean_dis', 'mean_len', 'mean_ang']
    range_len = [compare_ordered_json_AL(anns, preds, n, n + 10) for n in range(0, 100, 10)]
    range_len.append(compare_ordered_json_AL(anns, preds, 100, None))
    averg = np.average(np.array(range_len)[:, 2:5], 0)
    range_len.append(['-', '-', '-', '-', '-', '-'])
    range_len.append([None, None, 'averge_mdis', 'average_mlen', 'average_mang'])
    range_len.append([None, None, averg[0], averg[1], averg[2]])
    mean_dis, mean_len, mean_ang = compare_ordered_json(anns, preds)
    range_len.append(['-', '-', '-', '-', '-', '-'])
    range_len.append(['In total', None])
    range_len.append([None, None, np.mean(mean_dis), np.mean(mean_len), np.mean(mean_ang)])
    table = tabulate(range_len, headers, tablefmt="pipe")
    f = open(txt, 'a')
    print(txt, file=f)
    print(table, file=f)
    print(table)
    f.close()


# python tools/analyse_offset_omnicity.py
if __name__ == '__main__':
    pred_path = '/root/autodl-tmp/RBF-SAM-main/results/rbf_sam_omnicity/rbf_sam_omnicity.json'
    pred = json.load(open(pred_path))
    ann = COCO('/root/autodl-tmp/data/OmniCityView3WithOffset/coco/OmniCityView3WithOffset_test_roof.json')
    measure_detail(ann, pred,
                   os.path.join(
                       os.path.dirname(pred_path),
                       os.path.basename(pred_path).replace('.json', '.txt')
                   )
                   )
