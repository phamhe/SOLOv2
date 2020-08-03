import argparse
import os
import os.path as osp
import shutil
import tempfile
from scipy import ndimage
import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import init_dist, get_dist_info, load_checkpoint

from mmdet.core import coco_eval, results2json, wrap_fp16_model, tensor2imgs, get_classes
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
import cv2

import numpy as np
import matplotlib.cm as cm
import json
import os
import pickle
import copy 

def get_test_info(cfg, args):

    fin_ann_exp = open(cfg['ann_file'], 'r')
    new_ann = json.load(fin_ann_exp)
    new_ann['images'] = []
    new_ann['annotations'] = []
    imgs = [img for img in os.listdir(args.src_dir) if '.jpg' in img or '.png' in img]
    imgs.sort()
    for idx, img in enumerate(imgs):
        im = cv2.imread(os.path.join(args.src_dir, img))
        new_ann_img = {
            'file_name':img,
            'width':im.shape[1],
            'height':im.shape[1],
            'id':idx
        }
        new_ann_ann = {
            'segmentations':[],
            'area':0,
            'iscrowd':0,
            'image_id':idx,
            'bbox':[0, 0, 0, 0],
            'category_id':-1,
            'id':idx
        }
        new_ann['images'].append(new_ann_img)
        new_ann['annotations'].append(new_ann_ann)

    new_ann_file = os.path.join(args.out_dir, 'annos.json')
    fout_new_ann = open(new_ann_file, 'w')
    json.dump(new_ann, fout_new_ann)

    cfg['ann_file'] = new_ann_file
    cfg['img_prefix'] = args.src_dir

def vis_seg(data, result, img_norm_cfg, data_id, colors, score_thr, save_dir, type):
    img_tensor = data['img'][0]
    img_metas = data['img_meta'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
    assert len(imgs) == len(img_metas)
    if type=='cloth':
        class_names = get_classes('fashionpedia')
    else:
        class_names = get_classes('coco')

    mask_res = {}
    for img, img_meta, cur_result in zip(imgs, img_metas, result):
        mask_res['filename'] = img_meta['filename']
        mask_res['bboxes'] = []
        if cur_result is None:
            continue
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        
        seg_label = cur_result[0]
        seg_label_float = copy.deepcopy(cur_result[0])
        seg_label = seg_label.cpu().numpy().astype(np.uint8)
        seg_label_float = seg_label_float.cpu().numpy().astype(np.float32)

        cate_label = cur_result[1]
        cate_label = cate_label.cpu().numpy()

        score = cur_result[2].cpu().numpy()

        vis_inds = score > score_thr
        
        seg_label = seg_label[vis_inds]
        seg_label_float = seg_label_float[vis_inds]
        num_mask = seg_label.shape[0]
        cate_label = cate_label[vis_inds]
        cate_score = score[vis_inds]

        mask_density = []
        for idx in range(num_mask):
            cur_mask = seg_label[idx, :, :]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.int32)
            mask_density.append(cur_mask.sum())

        orders = np.argsort(mask_density)
        seg_label = seg_label[orders]
        seg_label_float = seg_label_float[orders]
        cate_label = cate_label[orders]
        cate_score = cate_score[orders]

        bboxes = []
        scores = []
        labels = []
        masks = []
        predicts = []
        for idx in range(num_mask):
            cur_score = cate_score[idx]
            cur_cate = cate_label[idx]
            label_text = class_names[cur_cate]

            idx = -(idx+1)
            cur_mask = seg_label[idx, :,:]
            cur_mask_float = seg_label_float[idx, :,:]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask_float = mmcv.imresize(cur_mask_float, (w, h))
            predicts.append(cur_mask_float)
            cur_mask = (cur_mask > 0.5).astype(np.uint8)

            x0 = 1000
            y0 = 1000
            x1 = -1
            y1 = -1
            for i in range(cur_mask.shape[0]):
                x = np.where(cur_mask[i, :]>0)[0]
                if len(x) > 0:
                    x0_tmp = x[0]
                    x1_tmp = x[-1]
                    if x0_tmp < x0:
                        x0 = x0_tmp
                    if x1_tmp > x1:
                        x1 = x1_tmp
                    if y0 == 1000:
                        y0 = i
                    y1 = i
                else:
                    continue
            masks.append(cur_mask)
            bboxes.append([x0, y0, x1, y1])
            scores.append(cur_score)
            labels.append(label_text)

        mask_res['mask'] = masks
        mask_res['predicts'] = predicts
        mask_res['bboxes'] = bboxes
        mask_res['scores'] = scores
        mask_res['labels'] = labels
        mask_res['ori_shape'] = img_meta['ori_shape']
        mask_res['img_shape'] = img_meta['img_shape']

    return mask_res

def single_gpu_test(model, data_loader, args, cfg=None, verbose=True):
    model.eval()
    results = []
    dataset = data_loader.dataset

    class_num = 1000 # ins
    colors = [(np.random.random((1, 3)) * 255).tolist()[0] for i in range(class_num)]    

    prog_bar = mmcv.ProgressBar(len(dataset))
    final_masks = {}
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            seg_result = model(return_loss=False, rescale=True, **data)
            result = None
        results.append(result)

        if verbose:
            mask = vis_seg(data, seg_result, cfg.img_norm_cfg, data_id=i, colors=colors, score_thr=args.score_thr, save_dir=args.out_dir, type=args.type)
            final_masks[mask['filename']] = mask

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results, final_masks


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
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
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
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


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--src_dir', help='input result file')
    parser.add_argument('--out_dir', help='output result file')
    parser.add_argument('--type', help='output result file type')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')

    # default setting
    parser.add_argument('--show', action='store_true', help='show results', default=1)
    parser.add_argument('--score_thr', type=float, default=0.3, help='score threshold for visualization')
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


def main():
    args = parse_args()

    assert args.out_dir, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out_dir"')

    #if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
    #    raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
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
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out_dir, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out_dir"')

    #if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
    #    raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    get_test_info(cfg.data.test, args)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    assert not distributed
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs, masks = single_gpu_test(model, data_loader, args, cfg=cfg)
    else:
        model = MMDistributedDataParallel(model.cuda())
        outputs = multi_gpu_test(model, data_loader, args.tmpdir)

    # save results
    outfile = os.path.join(args.out_dir, '%s_output.pkl'%(args.type))
    fout_output = open(outfile, 'wb')
    pickle.dump(masks, fout_output)

if __name__ == '__main__':
    main()
