import os.path as osp
import pickle
import shutil
import tempfile
import time
import os
import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results, tensor2imgs


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    # 如果指定了可视化输出目录，则创建
    if out_dir is not None:
        if not osp.exists(out_dir):
            os.makedirs(out_dir)
    # 设置模型为推理模式
    model.eval()
    # 用于存放所有图片的预测结果
    results = []
    # 获取数据集对象
    dataset = data_loader.dataset
    # 进度条
    prog_bar = mmcv.ProgressBar(len(dataset))
    # 遍历数据加载器
    for i, data in enumerate(data_loader):
        # if i>=6:
        #     continue
        # 如果存在 gt 框但没有标注，则跳过该样本
        gtbox = data.get('gt_bboxes', None)
        if gtbox is not None:
            if data['gt_bboxes'][0].shape[1]==0:
                continue

        # 前向推理，不计算梯度
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        # 如果需要可视化或保存可视化结果
        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    # img_meta,
                    # show=show,
                    out_file=out_file,
                    # score_thr=show_score_thr
                    )

        # -----------------------------
        # 根据模型输出的类型对结果进行编码/整理
        # result 的形式可能有几种：
        #  - (bbox_results, mask_results)                     -> 普通 Mask R-CNN 输出
        #  - (bbox_results, mask_results, offset_results)     -> Mask R-CNN + Offset
        #  - (bbox_results, mask_results, offset_results, _)  -> Mask R-CNN + Offset + Height（额外输出）
        # 所以根据 tuple 长度做不同的处理。
        # -----------------------------
        # encode mask results
        if isinstance(result, tuple) and len(result) == 2:
            # Mask R-CNN 的输出 (bbox_results, mask_results)
            # bbox_results: 通常是每张图的 bbox 预测结果，格式为每个类别一个 numpy 数组，
            #               每个数组形状为 (num_boxes, 5) -> [x1, y1, x2, y2, score]
            # mask_results: 与 bbox_results 对应的掩码预测（可能是二值 mask 列表或 ndarray）
            bbox_results, mask_results = result
            # 把 mask 从 raw 二值/ndarray 格式编码为 COCO 兼容的 RLE/字典格式，
            # 便于序列化保存（例如保存到 json 或 pkl，或用于 COCO API 评估）。
            encoded_mask_results = encode_mask_results(mask_results)
            result = bbox_results, encoded_mask_results
        elif isinstance(result, tuple) and len(result) == 3:
            # Mask R-CNN + Offset 的输出 (bbox_results, mask_results, offset_results)
            # 注意：有些模型会额外输出 offset（或其他自定义量），因此 tuple 长度为 3。
            bbox_results, mask_results, offset_results = result
            if mask_results is not None:
                # 若同时有 mask，则把 mask 编码为 RLE
                encoded_mask_results = encode_mask_results(mask_results)
                # 这里将 img_metas（data['img_metas'][0]）一并放入返回值里。
                # 这样做通常是为了后续处理（例如生成带元信息的结果或特殊评估）方便使用图像元信息。
                # 最终打包成 (bbox_results, img_metas, encoded_mask_results, offset_results)
                result = bbox_results, data['img_metas'][0], encoded_mask_results, offset_results
            else:
                # only pred offset
                # 如果没有 mask（mask_results 为 None），只返回 bbox + img_metas + offset
                # （不同的后续处理函数会根据返回结构区分如何处理）
                result = bbox_results, data['img_metas'][0], offset_results
        elif isinstance(result, tuple) and len(result) == 4:
            # Mask R-CNN + Offset + Height 的输出 (bbox_results, mask_results, offset_results, _)
            # 这个 branch 假定第四项是某些额外输出（比方说 height），但这里不使用它（用 _ 占位）
            bbox_results, mask_results, offset_results, robust_bbox_results = result
            # 同样对 mask 做编码
            encoded_mask_results = encode_mask_results(mask_results)
            # 打包为 (bbox_results, img_metas, encoded_mask_results, offset_results, None)
            # 最后 append 的结构有时需要固定字段数（例如便于下游解析），这里在末尾补了 None 占位。
            result = bbox_results, data['img_metas'][0], encoded_mask_results, offset_results, robust_bbox_results

        elif isinstance(result, tuple) and len(result) == 6:
            # Mask R-CNN + Offset + Height 的输出 (bbox_results, mask_results, offset_results, _)
            # 这个 branch 假定第四项是某些额外输出（比方说 height），但这里不使用它（用 _ 占位）
            bbox_results, mask_results, offset_results, offset_pred, offset_rf2ft, offset_ft2rf = result
            encoded_mask_results = encode_mask_results(mask_results)
            result = (bbox_results, data['img_metas'][0], encoded_mask_results, offset_results,
                      offset_pred, offset_rf2ft, offset_ft2rf)

        elif isinstance(result, tuple) and len(result) == 8:
            bbox_results, mask_results, offset_results, offset_global, offset_ll, offset_lh, offset_hl, offset_hh = result
            encoded_mask_results = encode_mask_results(mask_results)
            result = (bbox_results, data['img_metas'][0], encoded_mask_results, offset_results,
                      offset_global, offset_ll, offset_lh, offset_hl, offset_hh)

        elif isinstance(result, tuple) and len(result) == 9:
            bbox_results, mask_results, offset_results, offset_global, offset_ll, offset_lh, offset_hl, offset_hh, box_refine = result
            encoded_mask_results = encode_mask_results(mask_results)
            result = (bbox_results, data['img_metas'][0], encoded_mask_results, offset_results,
                      offset_global, offset_ll, offset_lh, offset_hl, offset_hh, box_refine)
        # -----------------------------
        # 将整理好的结果加入 results 列表
        # 注意：result 可能是按 batch 返回的“批内所有图片的结果”的集合（依赖 model 的实现）
        # 这里把整个 result 对象（可能包含多个图像的预测）作为一个元素 append，
        # 后续 mmdet 的汇总/格式化函数会据此将结果展开为每张图片对应的条目。
        # -----------------------------
        results.append(result)
        # -----------------------------
        # 更新进度条（按真实的图片数量更新）
        # data['img_metas'][0] 是一个 DataContainer，它的 .data 通常是一个 list，
        # 每个元素对应 batch 中的一张图片的 meta dict（例如 ori_shape, ori_filename 等）。
        # 所以 len(data['img_metas'][0].data) 就是当前 batch 的图片数（batch_size）。
        # 进度条需要按图片数量前进，而不是按 batch 数量。
        # -----------------------------
        batch_size = len(data['img_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()
        
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result, tuple) and len(result) == 2:
                bbox_results, mask_results = result
                encoded_mask_results = encode_mask_results(mask_results)
                result = bbox_results, encoded_mask_results
            elif isinstance(result, tuple) and len(result) == 3:
                bbox_results, mask_results, offset_results = result
                if mask_results is not None:
                    encoded_mask_results = encode_mask_results(mask_results)
                    result = bbox_results, encoded_mask_results, offset_results
                else:
                    # only pred offset
                    result = bbox_results, offset_results
            elif isinstance(result, tuple) and len(result) == 4:
                bbox_results, mask_results, offset_results, robust_bbox_results = result
                encoded_mask_results = encode_mask_results(mask_results)
                result = bbox_results, encoded_mask_results, offset_results, robust_bbox_results
            elif isinstance(result, tuple) and len(result) == 6:
                bbox_results, mask_results, offset_results, offset_pred, offset_rf2ft, offset_ft2rf = result
                encoded_mask_results = encode_mask_results(mask_results)
                result = (bbox_results, data['img_metas'][0], encoded_mask_results, offset_results,
                          offset_pred, offset_rf2ft, offset_ft2rf)
            elif isinstance(result, tuple) and len(result) == 8:
                bbox_results, mask_results, offset_results, offset_global, offset_ll, offset_lh, offset_hl, offset_hh = result
                encoded_mask_results = encode_mask_results(mask_results)
                result = (bbox_results, data['img_metas'][0], encoded_mask_results, offset_results,
                          offset_global, offset_ll, offset_lh, offset_hl, offset_hh)
            elif isinstance(result, tuple) and len(result) == 9:
                bbox_results, mask_results, offset_results, offset_global, offset_ll, offset_lh, offset_hl, offset_hh, box_refine = result
                encoded_mask_results = encode_mask_results(mask_results)
                result = (bbox_results, data['img_metas'][0], encoded_mask_results, offset_results,
                          offset_global, offset_ll, offset_lh, offset_hl, offset_hh, box_refine)
        results.append(result)

        if rank == 0:
            batch_size = (
                len(data['img_meta'].data)
                if 'img_meta' in data else len(data['img_metas'][0].data))
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
