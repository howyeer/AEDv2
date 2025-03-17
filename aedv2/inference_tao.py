# ------------------------------------------------------------------------
# Modified from MOTRv2 (https://github.com/megvii-research/MOTRv2)
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import json
import datetime
import os
import argparse
import torchvision.transforms.functional as F
import util.misc as utils
import torch
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from models.aedv2 import RuntimeTrackerBase
# from models import build
from groundingdino.models.registry import MODULE_BUILD_FUNCS
from datasets.taodataset import build as build_dataset
# from util.tool import load_model

from tao.toolkit.tao import Tao
from util.evaluation import teta_eval
from copy import deepcopy

from aedv2.models.structures import Instances
from torch.utils.data import Dataset, DataLoader
from engine import build_captions_and_token_span

import multiprocessing as mp

#--------grounding-dino------------
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap



class ListImgDataset(Dataset):
    def __init__(self, mot_path, img_list, det_db) -> None:
        super().__init__()
        self.mot_path = mot_path
        self.img_list = img_list
        self.det_db = det_db

        '''
        common settings
        '''
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_img_from_file(self, f_path):
        cur_img = cv2.imread(os.path.join(self.mot_path, f_path))
        assert cur_img is not None, f_path
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        proposals = []
        im_h, im_w = cur_img.shape[:2]
        for line in self.det_db[f_path[:-4] + '.txt']:
            l, t, w, h, s = list(map(float, line.split(',')))
            l = max(0, min(l, im_w - 1))
            t = max(0, min(t, im_h - 1))
            w = max(0, min(w, im_w - l))
            h = max(0, min(h, im_h - t))
            proposals.append([(l + w / 2) / im_w,
                                (t + h / 2) / im_h,
                                w / im_w,
                                h / im_h,
                                s])
        return cur_img, torch.as_tensor(proposals).reshape(-1, 5)

    def init_img(self, img, proposals):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img, proposals

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img, proposals = self.load_img_from_file(self.img_list[index])
        return self.init_img(img, proposals)

class MyTracker:
    def __init__(self, args, model, data):
        self.args = args
        self.detr = model
        self.imgs = data['imgs']
        self.img_infos = data['img_infos']
        self.ori_images = data['ori_imgs']
        cat_names_dict = data['cat_names']
        self.captions_dict = build_captions_and_token_span(cat_names_dict)
        self.seq_name = self.img_infos[0]['file_name'].split('/')[:3]

        self.img_len = len(self.imgs)
        self.result = list()

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]

    def detect(self, area_threshold=100):
        total_dts = 0
        track_instances = None
        for i, (cur_img, img_info, ori_img) in enumerate(zip(self.imgs, self.img_infos, self.ori_images)):
            image_id = img_info['image_id']
            video_id = img_info['video_id']
            # ori_img is PIL image
            ori_img_tensor = F.to_tensor(ori_img)
            cur_img = cur_img.unsqueeze(0)

            cur_img = cur_img.cuda()
            if track_instances is not None:
                track_instances.remove('boxes')
            _, seq_h, seq_w= ori_img_tensor.shape

            assert seq_h == img_info['height'] and seq_w == img_info['width']

            res = self.detr.inference_single_image(cur_img, (seq_h, seq_w), track_instances, self.captions_dict)
            track_instances = res['track_instances']
            num_active_proposals = res['num_active_proposals']

            dt_instances = deepcopy(track_instances[:num_active_proposals])
            dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)

            total_dts += len(dt_instances)

            bbox_xyxy = dt_instances.boxes.tolist()
            identities = dt_instances.obj_ids.tolist()
            labels = dt_instances.labels.tolist()
            scores = dt_instances.det_scores.tolist()
            track_ids = dt_instances.obj_ids.tolist()
            is_news = dt_instances.new.tolist()

            if len(labels) != 0:
                for xyxy, track_id, label, score, track_id, is_new in zip(bbox_xyxy, identities, labels, scores, track_ids, is_news):
                    if track_id < 0 or track_id is None:
                        raise ValueError('track_id < 0 or track_id is None')
                    x1, y1, x2, y2 = xyxy
                    w, h = x2 - x1, y2 - y1
                    self.result.append({
                        'image_id': image_id,
                        'category_id': label,
                        'bbox': [x1, y1, w, h],
                        'score': score,
                        'track_id': track_id,
                        'video_id': video_id,
                        'is_new': is_new,
                    })
        return self.result, total_dts

def load_infer_model(model_config_path, model_checkpoint_path, cpu_only=False):
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    args.device = "cuda" if not cpu_only else "cpu"
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion = build_func(args)
    model = model.to(args.device)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    _ = criterion.eval()
    
    return model, criterion



if __name__ == '__main__':
    parser = argparse.ArgumentParser('inference_tao', add_help=True)
    parser.add_argument('--miss_tolerance', default=10, type=int)
    parser.add_argument('--exp_name', default='tracker', type=str)
    parser.add_argument('--split', default='val', type=str, choices=['val', 'test'])
    #------------------------grounding-dino--------------------------------
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="aedv2/output_dir", help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")

    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    cfg = SLConfig.fromfile(args.config_file)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))
    args.with_box_refine = False
    print(args)
    # load model and weights
    # detr, _, _ = build_model(args)
    model, criterion = load_infer_model(cfg, checkpoint_path)
    model.track_base = RuntimeTrackerBase(args.val_match_high_thresh, args.val_match_low_thresh, args.miss_tolerance, args.match_high_score)

    dataset_val = build_dataset(image_set = 'val', args=args)
    collate_fn = utils.mot_collate_fn
    data_loader_val = DataLoader(dataset_val, collate_fn=collate_fn, num_workers=args.num_workers, batch_size=1,
                                   shuffle=False, drop_last=False, pin_memory=True)

    result = list()
    pbar = tqdm(data_loader_val)
    for data in pbar:
        tracker = MyTracker(args, model=model, data=data)
        result_i, total_dts = tracker.detect()
        pbar.set_description("{} dts in total".format(total_dts))
        result.extend(result_i)

    # save and evaluate
    predict_path = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(predict_path, exist_ok=True)
    # add date and time
    result_name = 'infer_result' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.json'
    result_path = os.path.join(predict_path, result_name)
    print('saving inference result to {}'.format(result_path))
    json.dump(result, open(result_path, 'w'), indent=4)







def sub_processor(pid, seq_nums, args):
    # load model and weights
    gpu_id = pid % torch.cuda.device_count()
    torch.cuda.set_device(gpu_id)
    detr, _, _ = build_model(args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    # checkpoint_id = int(args.resume.split('/')[-1].split('.')[0].split('_')[-1])
    checkpoint_id = int(args.resume.split('/')[-1].split('.')[0].split('t')[-1])
    detr = load_model(detr, args.resume)
    detr.eval()
    detr = detr.cuda()

    device = args.device
    detr.to(device)
    for seq_num in seq_nums:
        print('Evaluating seq {}'.format(seq_num))
        det = Detector(args, checkpoint_id, model=detr, seq_num=seq_num)
        det.detect()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    expressions_root = os.path.join(args.rmot_path, 'expression')
    if "refer_kitti-v2" in args.rmot_path:
        video_ids = ['0005', '0011', '0013','0019']
    else:
        video_ids = ['0005', '0011', '0013']


    seq_nums = []
    for video_id in video_ids:  # we have multiple videos
        expression_jsons = sorted(os.listdir(os.path.join(expressions_root, video_id)))
        for expression_json in expression_jsons:  # each video has multiple expression json files
            seq_nums.append([video_id, expression_json])

    thread_num = 9
    processes = []
    expression_num = len(seq_nums)
    per_thread_seq_num = expression_num // thread_num

    print("Start inference")
    for i in range(thread_num):
        if i == thread_num -1:
            sub_seq_list = seq_nums[i*per_thread_seq_num:]
        else:
            sub_seq_list = seq_nums[i*per_thread_seq_num:(i+1)*per_thread_seq_num]
        p = mp.Process(target=sub_processor,args=(i, sub_seq_list, args))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    print("Over")