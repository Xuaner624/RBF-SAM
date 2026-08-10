import argparse
import os
import sys

sys.path.append("./")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# import tools.fuse_conv_bn.fuse_module as fuse_module
custom_mmdet_root = "/root/autodl-tmp/RBF-SAM-main"

if custom_mmdet_root not in sys.path:
    sys.path.insert(0, custom_mmdet_root)
    print(f"[INFO] Using custom mmdet from: {custom_mmdet_root}")

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from tools.fuse_conv_bn import fuse_module

    
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
import json
from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')

    parser.add_argument('--config',
                        default="configs/rbf_sam/rbf_sam_bonai.py",
                        help='test config file path')

    parser.add_argument('--checkpoint',
                        default='results/rbf_sam_bonai/rbf_sam_bonai_epoch_48.pth',
                        help='checkpoint file')

    parser.add_argument('--out', help='output result file in pickle format')

    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')

    parser.add_argument(
        '--format-only',
        default=True,
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')

    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
             ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')

    parser.add_argument('--show', action='store_true', help='show results')

    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')

    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')

    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')

    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu-collect is not specified')

    parser.add_argument(
        '--options',
        default=dict(jsonfile_prefix='./mask_rcnn_test-dev_results'),
        nargs='+', action=DictAction, help='arguments in dict')

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')

    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


js = json.load(open('/root/autodl-tmp/data/BONAI/coco/bonai_shanghai_xian_test.json'))
# js = json.load(open('D:/off_nadir_building/data/BONAI/coco/bonai_shanghai_xian_test.json'))


def search_img_id(name, anns=js['images']):
    for ann in anns:
        if name == ann['file_name']:
            return ann['id']


def search_ann_id(bbox, img_id, key, coco=COCO('/root/autodl-tmp/data/BONAI/coco/bonai_shanghai_xian_test.json')):
                  # coco=COCO('D:/off_nadir_building/data/BONAI/coco/bonai_shanghai_xian_test.json')):
    anns = coco.getAnnIds(img_id)
    # max_iou=0
    # ann_id = None
    for ann in anns:
        anninfo = coco.loadAnns(ann)[0]
        x, y = anninfo[key][0], anninfo[key][1]
        x = x if x > 0 else 0
        y = y if y > 0 else 0
        if x == bbox[0]:
            if y == bbox[1]:
                return anninfo['id']
    return None


def main():
    args = parse_args()
    assert args.out or args.eval or args.format_only or args.show \
           or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    print(f'using config {args.config}')
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    cfg.data.test.test_mode = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    json_ann = []
    coco = COCO(cfg.data.test.ann_file)
    js = json.load(open(cfg.data.test.ann_file))

    if len(outputs[0]) == 4:
        for i, (building_bbox, img_matas, masks, offsets) in enumerate(outputs):
            for box, mask, offset in zip(building_bbox, masks, offsets):
                img_id = search_img_id(img_matas.data[0][0]["ori_filename"], js['images'])
                box[2] = box[2] - box[0]
                box[3] = box[3] - box[1]
                key = 'building_bbox'  # 'roof_bbox' if 'roof' in os.path.basename(args.config) else
                id = search_ann_id(box, img_id, key=key, coco=coco)
                json_ann.append({"building_bbox": box[:4].tolist(),
                                 "offset": offset.tolist(),
                                 "file_name": img_matas.data[0][0]["ori_filename"],
                                 'image_id': img_id,
                                 'id': id,
                                 'segmentation': mask[0],
                                 'roof_mask': mask[0],
                                 'building_seg': mask[1],
                                 'category_id': 1,
                                 'score': 1.0,
                                 })

    if len(outputs[0]) == 3:
        for i, (building_bbox, img_matas, offsets) in enumerate(outputs):
            for box, offset in zip(building_bbox, offsets):
                img_id = search_img_id(img_matas.data[0][0]["ori_filename"], js['images'])
                key = 'building_bbox'  # 'roof_bbox' if 'roof' in os.path.basename(args.config) else
                id = search_ann_id(box, img_id, key, coco=coco)
                json_ann.append({"building_bbox": box[:4].tolist(),
                                 "offset": offset.tolist(),
                                 "file_name": img_matas.data[0][0]["ori_filename"],
                                 'image_id': img_id,
                                 'id': id,
                                 'category_id': 1,
                                 'score': 1.0,
                                 })

    if len(outputs[0]) == 5:
        for i, (building_bbox, img_matas, masks, offsets, _) in enumerate(outputs):
            for box, offset, mask in zip(building_bbox, offsets, masks):
                img_id = search_img_id(img_matas.data[0][0]["ori_filename"], js['images'])
                key = 'roof_bbox' if 'roof' in os.path.basename(args.config) else 'building_bbox'
                id = search_ann_id(box, img_id, key, coco=coco)
                json_ann.append({"building_bbox": box[:4].tolist(),
                                 "offset": offset.tolist(),
                                 "file_name": img_matas.data[0][0]["ori_filename"],
                                 'image_id': img_id,
                                 'roof_mask': mask[0],
                                 'id': id,
                                 'category_id': 1,
                                 'score': 1.0,
                                 })

    if len(outputs[0]) == 10:
        for i, (building_bbox, img_matas, masks, offsets, offset_global, offset_ll, offset_lh, offset_hl, offset_hh, box_robust) \
                in enumerate(outputs):
            for box, offset, mask, offset_global, offset_ll, offset_lh, offset_hl, offset_hh, box_robust \
                    in zip(building_bbox, offsets, masks, offset_global, offset_ll, offset_lh, offset_hl, offset_hh, box_robust):
                img_id = search_img_id(img_matas.data[0][0]["ori_filename"], js['images'])
                box[2] = box[2] - box[0]
                box[3] = box[3] - box[1]
                key = 'roof_bbox' if 'roof' in os.path.basename(args.config) else 'building_bbox'
                id = search_ann_id(box, img_id, key, coco=coco)
                # "building_bbox": Ground truth building box for subsequent evaluation
                # "robust_bbox": Input box prompt (supported types: gt, building footprint, roof, noisy)
                json_ann.append({"building_bbox": box[:4].tolist(),
                                 "robust_bbox": box_robust[:4].tolist(),
                                 "offset": offset.tolist(),
                                 "offset_global": offset_global.tolist(),
                                 "offset_ll": offset_ll.tolist(),
                                 "offset_lh": offset_lh.tolist(),
                                 "offset_hl": offset_hl.tolist(),
                                 "offset_hh": offset_hh.tolist(),
                                 "file_name": img_matas.data[0][0]["ori_filename"],
                                 'image_id': img_id,
                                 'roof_mask': mask[0],
                                 'id': id,
                                 'category_id': 1,
                                 'score': 1.0,
                                 })

    save_dir = os.path.abspath('./results/rbf_sam_bonai')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(args.config).replace('.py', '.json'))
    mmcv.dump(json_ann, save_path)
    print(f"The result has been saved to: {save_path}")


# python tools/test_rbf_sam_bonai.py --config configs/rbf_sam/rbf_sam_bonai.py
if __name__ == '__main__':
    main()
