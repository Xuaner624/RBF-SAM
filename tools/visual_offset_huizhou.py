import json
import numpy as np
import cv2
import sys
import math
from os.path import join as ospj
import os

sys.path.append("./")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# import tools.fuse_conv_bn.fuse_module as fuse_module
custom_mmdet_root = "/root/autodl-tmp/RBF-SAM-main"

if custom_mmdet_root not in sys.path:
    sys.path.insert(0, custom_mmdet_root)
    print(f"[INFO] Using custom mmdet from: {custom_mmdet_root}")

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from multiprocessing import Pool
from mmdet.utils.boundary_iou import mask_to_boundary
from tqdm import tqdm
import multiprocessing


def decode_rle(ann):
    rle = [ann] if not isinstance(ann, list) else ann
    rle = maskUtils.merge(rle)
    binary_mask = maskUtils.decode(rle)
    return binary_mask


def decode_polygon(ann):
    rle = maskUtils.frPyObjects(ann, 1024, 1024)
    rle = maskUtils.merge(rle)
    binary_mask = maskUtils.decode(rle)
    return binary_mask


def save_json(save_path, data):
    assert save_path.split('.')[-1] == 'json'
    with open(save_path, 'w') as file:
        json.dump(data, file)


SAVE_PATH = '/root/autodl-tmp/RBF-SAM-main/results/rbf_sam_huizhou_vis'
ANN_COCO = COCO(
    '/root/autodl-tmp/data/BONAI/coco/building_seg_obm_huizhou_test.json',)

def vis_boundary_offset(coco_path, save_path):
    coco = COCO(coco_path) if isinstance(json.load(open(coco_path)), dict) else None
    if not coco:
        print("The input file is not in standard COCO format.")
        js_anns = json.load(open(coco_path))
        imgs = []
        categories = [{'id': 1, 'name': 'building', 'supercategory': 'building'}]
        for ann in js_anns:
            img = dict(file_name=ann['file_name'],
                       id=ann['image_id'],
                       height=1024,
                       width=1024)
            imgs.append(img)

        coco = dict(images=imgs, annotations=js_anns, categories=categories)
        save_json('tmp.json', coco)
        coco = COCO('tmp.json')

    imgids = coco.getImgIds()
    [_draw_boundary_offset(id, coco, save_path) for id in tqdm(imgids)]


def _draw_boundary_offset(id, coco: COCO, save_dir=SAVE_PATH):
    if not os.path.exists(save_dir):
        os.system('mkdir {}'.format(save_dir))
    img = coco.loadImgs(id)[0]
    annIds = coco.getAnnIds(id)
    anns = coco.loadAnns(annIds)
    offsets = [ann['offset'] for ann in anns]
    offsets = np.array(offsets)
    """
    example：
    "offset": [
        1.449838187702269,
        -10.280906148867075
    ]
    """
    offsets_len = np.sqrt(np.sum(offsets * offsets, axis=1))
    max_offset = np.max(offsets_len)
    offsets_gray = offsets_len / max_offset * 230
    segs = np.zeros((1024, 1024))

    buildings = []
    foot_segs = np.zeros((1024, 1024))
    image = cv2.imread(os.path.join('/root/autodl-tmp/data/BONAI/huizhou_test', img['file_name']))

    # for each building
    for i, ann in enumerate(anns):
        roof_seg = ann['roof_mask']
        seg = decode_rle(roof_seg) if isinstance(roof_seg, dict) else decode_polygon([roof_seg])
        seg = mask_to_boundary(seg, 0.002)
        pred_offset = np.array(ann['offset'])
        """
        example：
        "roof_bbox": [
            10.0,
            807.0,
            100.0,
            108.0
        ],
        """
        # draw building
        segs[seg > 0] = seg[seg > 0]
        translation_matrix = np.float32([[1, 0, -pred_offset[0]], [0, 1, -pred_offset[1]]])
        seg_ = cv2.warpAffine(seg, translation_matrix, (1024, 1024))
        foot_segs[seg_ > 0] = seg_[seg_ > 0]

    image[foot_segs > 0, 0] = 0
    image[foot_segs > 0, 1] = 255
    image[foot_segs > 0, 2] = 255

    image[segs > 0, 0] = 255
    image[segs > 0, 1] = 255
    image[segs > 0, 2] = 0
    cv2.imwrite(ospj(save_dir, img['file_name']), image.astype('uint8'))

    result = image.copy()
    # 保存结果
    cv2.imwrite(ospj(save_dir, img['file_name']), result)


# python tools/visual_offset_huizhou.py
if __name__ == '__main__':
    vis_boundary_offset('/root/autodl-tmp/RBF-SAM-main/results/rbf_sam_huizhou/rbf_sam_huizhou.json',
                        '/root/autodl-tmp/RBF-SAM-main/results/rbf_sam_huizhou_vis')
