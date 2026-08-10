import os
import os.path as osp
import math
import tempfile
import csv
import numpy as np

from collections import defaultdict

from .coco import CocoDataset
from .builder import DATASETS


# 将数据集注册到框架的 DATASETS 中，方便调用
@DATASETS.register_module()
class BONAI_OBM(CocoDataset):
    # 数据集类别，这里只有一个类别：building（建筑）
    CLASSES = ('building')

    def __init__(self,
                 ann_file,  # 标注文件路径
                 pipeline,  # 数据处理流水线
                 classes=None,  # 类别
                 data_root=None,  # 数据根目录
                 img_prefix='',  # 图像路径前缀
                 seg_prefix=None,  # 分割标注路径前缀
                 edge_prefix=None,  # 建筑边缘标注路径前缀
                 side_face_prefix=None,  # 建筑侧面标注路径前缀
                 offset_field_prefix=None,  # 偏移场标注路径前缀
                 proposal_file=None,  # 候选框文件
                 test_mode=False,  # 是否测试模式
                 filter_empty_gt=True,  # 是否过滤没有标注的图片
                 gt_footprint_csv_file=None,  # 建筑物 footprint 标注文件
                 bbox_type='roof',  # bbox 类型（roof/building/footprint）
                 mask_type='roof',  # mask 类型（roof/footprint/building）
                 offset_coordinate='rectangle',  # 偏移坐标类型（矩形/极坐标）
                 resolution=0.6,  # 图像分辨率
                 ignore_buildings=True):  # 是否忽略 crowd 建筑

        # 调用父类初始化
        super(BONAI_OBM, self).__init__(ann_file=ann_file,
                                        pipeline=pipeline,
                                        classes=classes,
                                        data_root=data_root,
                                        img_prefix=img_prefix,
                                        seg_prefix=seg_prefix,
                                        proposal_file=proposal_file,
                                        test_mode=test_mode,
                                        filter_empty_gt=filter_empty_gt)
        # 保存一些自定义参数
        self.ann_file = ann_file
        self.bbox_type = bbox_type
        self.mask_type = mask_type
        self.offset_coordinate = offset_coordinate
        self.resolution = resolution
        self.ignore_buildings = ignore_buildings
        self.gt_footprint_csv_file = gt_footprint_csv_file

        # 额外的标注前缀路径
        self.edge_prefix = edge_prefix
        self.side_face_prefix = side_face_prefix
        self.offset_field_prefix = offset_field_prefix

        # 如果 data_root 有值
        if self.data_root is not None:
            # 若 edge_prefix 存在且不是绝对路径，则将其与 data_root 拼接构成绝对路径
            if not (self.edge_prefix is None or osp.isabs(self.edge_prefix)):
                self.edge_prefix = osp.join(self.data_root, self.edge_prefix)

        if self.data_root is not None:
            if not (self.side_face_prefix is None or osp.isabs(self.side_face_prefix)):
                self.side_face_prefix = osp.join(self.data_root, self.side_face_prefix)

        if self.data_root is not None:
            if not (self.offset_field_prefix is None or osp.isabs(self.offset_field_prefix)):
                self.offset_field_prefix = osp.join(self.data_root, self.offset_field_prefix)

        # print("This dataset has these keys: {}".format(list(self.get_properties(0))))
        # 用于打印数据集第 0 个样本的属性键，展示数据集的属性信息

    def pre_pipeline(self, results):
        """
        预处理 pipeline 的方法，用于在数据处理开始前
        给 results 字典增加一些必要的字段。
        - 调用父类的 pre_pipeline 方法，保持继承逻辑
        - 给 results 增加与边缘（edge）、侧面（side_face）、
          和偏移量（offset_field）相关的前缀和空字段列表，
          以便后续数据处理阶段填充。
        """
        super(BONAI_OBM, self).pre_pipeline(results)
        results['edge_prefix'] = self.edge_prefix
        results['edge_fields'] = []

        results['side_face_prefix'] = self.side_face_prefix
        results['side_face_fields'] = []

        results['offset_field_prefix'] = self.offset_field_prefix
        results['offset_field_fields'] = []

    def get_properties(self, idx):
        """
            根据指定索引，获取图像的标注属性字段。
        """
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)

        return ann_info[0].keys()

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.data_infos):
            img_id = img_info['id']
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            ann_info = self.coco.loadAnns(ann_ids)
            # 判断是否全为 crowd 标注
            all_iscrowd = all([_['iscrowd'] for _ in ann_info])
            if self.filter_empty_gt and (self.img_ids[i] not in ids_with_ann
                                         or all_iscrowd):
                continue
            # 如果图像的宽和高都大于等于最小尺寸阈值，则保留该图像索引
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        # 初始化各种 ground truth 容器
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_roof_masks_ann = []
        gt_building_masks_ann = []
        gt_footprint_masks_ann = []
        gt_offsets = []
        gt_building_heights = []
        gt_angles = []
        gt_mean_angle = 0.0
        gt_roof_bboxes = []
        gt_footprint_bboxes = []
        gt_only_footprint_flag = 0

        # 遍历标注信息,# 跳过 ignore 标注
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            # bbox type may be roof, building and footprint, you need to set the value in config file
            # 根据配置选择 bbox 类型 config中选用的是building，即整个building的框
            if self.bbox_type == 'roof':
                x1, y1, w, h = ann['bbox']
            elif self.bbox_type == 'building':
                x1, y1, w, h = ann['building_bbox']
            elif self.bbox_type == 'footprint':
                x1, y1, w, h = ann['footprint_bbox']
            else:
                raise (TypeError(f"don't support bbox_type={self.bbox_type}"))

            # 过滤无效 bbox（超出图像范围、面积为0、宽高过小、类别不在 cat_ids 中）
            # 计算候选框在图像宽度方向上的有效交集宽度
            # max(0, ...) 确保结果不为负数
            # min(x1 + w, img_info['width'])：框的右边界与图像右边界取最小值
            # max(x1, 0)：框的左边界与图像左边界取最大值
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            # 如果该目标被标记为 crowd 且需要忽略，则加入 ignore 列表
            if ann.get('iscrowd', False) and self.ignore_buildings:
                gt_bboxes_ignore.append(bbox)
            else:
                if 'roof_bbox' in ann:
                    x1, y1, w, h = ann['roof_bbox']
                    gt_roof_bboxes.append([x1, y1, x1 + w, y1 + h])
                if 'footprint_bbox' in ann:
                    x1, y1, w, h = ann['footprint_bbox']
                    gt_footprint_bboxes.append([x1, y1, x1 + w, y1 + h])
                if 'only_footprint' in ann:
                    if ann['only_footprint'] == 1:
                        gt_only_footprint_flag = 1
                    else:
                        gt_only_footprint_flag = 0

                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                # gt_only_footprint_flag=0: use roof as mask, gt_only_footprint_flag=0:use footprint as mask

                # 根据 mask_type 选择 mask，config中选用的是roof
                if gt_only_footprint_flag == 0:
                    if self.mask_type == 'roof':
                        gt_masks_ann.append(ann['segmentation'])
                    elif self.mask_type == 'footprint':
                        gt_masks_ann.append([ann['footprint_mask']])
                    elif self.mask_type == 'building':
                        gt_masks_ann.append(ann['building_seg'])
                    else:
                        raise (TypeError(f"don't support mask_type={self.mask_type}"))
                else:
                    gt_masks_ann.append([ann['footprint_mask']])

                gt_roof_masks_ann.append(ann['segmentation'])
                gt_footprint_masks_ann.append([ann['footprint_mask']])
                gt_building_masks_ann.append(ann['building_seg'])
                # rectangle coordinate -> offset = (x, y), polar coordinate -> offset = (length, theta)
                # 解析 offset（矩形坐标系 / 极坐标系）
                if 'offset' in ann:
                    if self.offset_coordinate == "rectangle":
                        gt_offsets.append(ann['offset'])
                    elif self.offset_coordinate == 'polar':
                        offset_x, offset_y = ann['offset']
                        length = math.sqrt(offset_x ** 2 + offset_y ** 2)
                        angle = math.atan2(offset_y, offset_x)
                        gt_offsets.append([length, angle])
                    else:
                        raise (RuntimeError(f'do not support this coordinate: {self.offset_coordinate}'))
                else:
                    gt_offsets.append([0, 0])

                # 建筑高度
                if 'building_height' in ann:
                    gt_building_heights.append(ann['building_height'])
                else:
                    gt_building_heights.append(0.0)

                # 角度计算（结合 offset 和高度）
                if 'offset' in ann and 'building_height' in ann:
                    offset_x, offset_y = ann['offset']
                    height = ann['building_height']
                    angle = math.atan2(math.sqrt(offset_x ** 2 + offset_y ** 2) * self.resolution, height)

                    gt_angles.append(angle)

        if gt_bboxes:
            # 如果存在有效标注，则将各类标注转换为 numpy 数组
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_roof_bboxes = np.array(gt_roof_bboxes, dtype=np.float32)
            gt_footprint_bboxes = np.array(gt_footprint_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_offsets = np.array(gt_offsets, dtype=np.float32)
            gt_building_heights = np.array(gt_building_heights, dtype=np.float32)
            gt_mean_angle = float(np.array(gt_angles, dtype=np.float32).mean())
            gt_only_footprint_flag = float(gt_only_footprint_flag)
        else:
            # 没有有效标注时，初始化为空数组或默认值
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_roof_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_footprint_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_offsets = np.zeros((0, 2), dtype=np.float32)
            gt_building_heights = np.zeros((0, 2), dtype=np.float32)
            gt_mean_angle = 0.0001
            gt_only_footprint_flag = 0

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        # 生成与图像文件对应的 mask 和辅助文件名
        seg_map = img_info['filename'].replace('jpg', 'png')
        edge_map = img_info['filename'].replace('jpg', 'png')
        side_face_map = img_info['filename'].replace('jpg', 'png')
        offset_field = img_info['filename'].replace('png', 'npy')
        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            roof_masks=gt_roof_masks_ann,
            footprint_masks=gt_footprint_masks_ann,
            building_masks=gt_building_masks_ann,
            seg_map=seg_map,
            offsets=gt_offsets,
            building_heights=gt_building_heights,
            angle=gt_mean_angle,
            edge_map=edge_map,
            side_face_map=side_face_map,
            roof_bboxes=gt_roof_bboxes,
            footprint_bboxes=gt_footprint_bboxes,
            offset_field=offset_field,
            only_footprint_flag=gt_only_footprint_flag)

        return ann

    def _segm2json(self, results):
        """
        将检测和分割结果转换为 COCO 格式的 JSON 可用数据。
        包括边界框结果 (bbox_json_results) 和分割结果 (segm_json_results)。
        """
        bbox_json_results = []
        segm_json_results = []
        # 遍历数据集中所有图片
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            # 根据结果长度，解包不同的预测信息
            if len(results[idx]) == 2:
                det, seg = results[idx]
            elif len(results[idx]) == 3:
                det, seg, offset = results[idx]
            elif len(results[idx]) == 4:
                det, seg, offset, building_height = results[idx]
            else:
                raise (RuntimeError("do not support the length of results: ", len(results[idx])))

            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)

        return bbox_json_results, segm_json_results

    def write_results2csv(self, results, meta_info=None):
        print("meta_info: ", meta_info)
        segmentation_eval_results = results[0]
        with open(meta_info['summary_file'], 'w') as summary:
            csv_writer = csv.writer(summary, delimiter=',')
            csv_writer.writerow(['Meta Info'])
            csv_writer.writerow(['model', meta_info['model']])
            csv_writer.writerow(['anno_file', meta_info['anno_file']])
            csv_writer.writerow(['gt_roof_csv_file', meta_info['gt_roof_csv_file']])
            csv_writer.writerow(['gt_footprint_csv_file', meta_info['gt_footprint_csv_file']])
            csv_writer.writerow(['vis_dir', meta_info['vis_dir']])
            csv_writer.writerow([''])
            for mask_type in ['roof', 'footprint']:
                csv_writer.writerow([mask_type])
                csv_writer.writerow([segmentation_eval_results[mask_type]])
                csv_writer.writerow(['F1 Score', segmentation_eval_results[mask_type]['F1_score']])
                csv_writer.writerow(['Precision', segmentation_eval_results[mask_type]['Precision']])
                csv_writer.writerow(['Recall', segmentation_eval_results[mask_type]['Recall']])
                csv_writer.writerow(['True Positive', segmentation_eval_results[mask_type]['TP']])
                csv_writer.writerow(['False Positive', segmentation_eval_results[mask_type]['FP']])
                csv_writer.writerow(['False Negative', segmentation_eval_results[mask_type]['FN']])
                csv_writer.writerow([''])

            csv_writer.writerow([''])

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by \
                piepline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)
