import torch
import torch.nn as nn
import copy
import sys
import cv2
import random
from tqdm import tqdm
import time
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import os.path as osp
import numpy as np
import json
import groundingdino.datasets.transforms as T

from tao.toolkit.tao import Tao
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image
from groundingdino.util.misc import collate_fn
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap, get_phrases_from_posmap_tao
# from models.structures import Instances
from collections import defaultdict
from torchvision.ops import nms
# from util.box_ops import box_cxcywh_to_xyxy, box_iou
# from util.misc import linear_assignment
from random import choice, randint
import argparse
import matplotlib.pyplot as plt

def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

def get_grounding_output(outputs, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    logits_filt  = outputs["pred_logits"].sigmoid()  # (bs,num_q, 256)
    boxes_filt = outputs["pred_boxes"]  # (bs,num_q, 4)

    # filter output
    if token_spans is None:
        # logits_filt = logits.cpu().clone()
        # boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=2)[0] > box_threshold   # [0]为返回值，max会返回values和indices
        logits_filt = [logits[mask] for logits, mask in zip(logits_filt, filt_mask)] #num_filt, 256 的list  num_filt不同
        boxes_filt = [boxes[mask] for boxes, mask in zip(boxes_filt, filt_mask)] #num_filt, 4 的list

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for fi in range(len(logits_filt)):
            pred_phrases_b = []
            for logit in logits_filt[fi]:
                pred_phrase = get_phrases_from_posmap_tao(logit, logit > text_threshold, tokenized, tokenlizer)
                if with_logits:
                    pred_phrases_b.append((pred_phrase, float(logit.max().item())))
                else:
                    pred_phrases_b.append(pred_phrase)
            pred_phrases.append(pred_phrases_b)
    return boxes_filt, pred_phrases

class TAODataset(Dataset):
    def __init__(self, tao_root: str, ann_file:str,logger=None, transform=None, one_class=True):
        super().__init__()
        self.root = tao_root
        self.logger = logger
        assert ann_file in ['train', 'val', 'test']
        if ann_file == 'train':
            annotation_path = osp.join(self.root, 'annotations', 'train_ours_v1.json')
        elif ann_file == 'val':
            annotation_path = osp.join(self.root, 'annotations', 'validation_ours_v1')
        elif ann_file == 'test':
            annotation_path = osp.join(self.root, 'annotations', 'tao_test_burst_v1.json')
        print('using annotation path {}'.format(annotation_path))
        tao = Tao(annotation_path, self.logger)
        self.tao = tao
        self.transform = transform
        self.id_list = tao.get_img_ids()
        self.cat_list = [item['name'] for item in self.tao.cats.values()]
        
    
    def __getitem__(self, idx):
        img_info = self.tao.load_imgs([self.id_list[idx]])
        ori_img = Image.open(osp.join(self.root, 'frames', img_info[0]['file_name']))
        target = self.tao.img_ann_map[self.id_list[idx]]

        w, h = ori_img.size
        boxes = [obj["bbox"] for obj in target]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        # filt invalid boxes/masks/keypoints
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        target_new = {}
        target_new["image_id"] = self.id_list[idx]
        target_new["boxes"] = boxes
        target_new["ori_size"] = torch.as_tensor([int(h), int(w)])
        target_new["category_id"] = [obj["category_id"] for obj in target]
        target_new["video_id"] = [img_info[0]['video_id']]
        target_new["cat_id_dic"] = {}
        for obj in target:
            cat_str = self.cat_list[obj["category_id"]-1]
            cat_str = cat_str.split("(")[0].replace("_", " ").strip()
            cat_str = cat_str.lower()
            target_new["cat_id_dic"][cat_str] = obj["category_id"] 

        if self.transform is not None:
            img, target = self.transform(ori_img, target_new)
        
        return img, target
    
    def __len__(self):
        return len(self.tao.img_ann_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TAO dataset loading", add_help=False)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=1, type=int)

    # load model
    parser.add_argument("--config_file", "-c", type=str, default="groundingdino/config/GroundingDINO_SwinT_OGC.py",
                        help="path to config file")
    parser.add_argument("--checkpoint_path", "-p", type=str, default = "weights/groundingdino_swint_ogc.pth",
                        help="path to checkpoint file"
    )
    parser.add_argument("--device", type=str, default="cuda",
                        help="running device (default: cuda)")

    # post processing
    parser.add_argument("--num_select", type=int, default=300,
                        help="number of topk to select")

    # data info
    parser.add_argument("--tao_root", type=str, default='/data/fzm_2022/Datasets/TAO',
                        help="coco root")
    parser.add_argument("--ann_file", type=str, default='test',
                        help="train or val or test")
    # threshold
    parser.add_argument("--box_threshold", type=float, default=0.1, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.08, help="text threshold")

    # save_path
    parser.add_argument("--output_json", type=str, default='output/tao_det')
    args = parser.parse_args()

    # ann_path = '/data/fzm_2022/Datasets/TAO/annotations/train_ours_v1.json'
    # tao = Tao(ann_path)
    # cats = tao.cats
    # print(len(cats.keys()))

    # config
    cfg = SLConfig.fromfile(args.config_file)

    # build model
    model = load_model(args.config_file, args.checkpoint_path)
    model = model.to(args.device)
    model = model.eval()
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = TAODataset(args.tao_root, args.ann_file, logger=None, transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # build captions
    category_dict = dataset.tao.cats
    cat_list = [item['name'] for item in category_dict.values()]
    # caption = " . ".join(cat_list) + ' .'
    # caption = caption.lower()
    # caption = caption.strip()
    # print("Input text prompt:", caption)

    # run inference
    res_json_list = []
    with_logits = True
    start = time.time()
    for i, (images, targets) in enumerate(tqdm(data_loader)):
        # get images and captions
        images = images.tensors.to(args.device)
        bs = images.shape[0]

        # get captions
        img_cat_dic = dict()
        for target in targets:
            img_cat_dic.update(target['cat_id_dic'])
        img_cat_list = img_cat_dic.keys()
        caption = " . ".join(img_cat_list) + ' .'
        caption = caption.lower()
        caption = caption.strip()
        # print("Input text prompt:", caption)
        input_captions = [caption] * bs

        # feed to the model
        outputs = model(images, captions=input_captions)

        
        boxes_filt_list, pred_phrases_list = get_grounding_output(outputs, caption, args.box_threshold, args.text_threshold, with_logits=with_logits)

        for j in range(bs):
            boxes = boxes_filt_list[j].to(images.device)
            pred_phrases = pred_phrases_list[j]
            assert len(boxes) == len(pred_phrases), "boxes and labels must have same length"
            orig_target_sizes = targets[j]['ori_size'].to(images.device)

            assert orig_target_sizes.shape[0] == 2

            img_h, img_w = orig_target_sizes
            scale_fct = torch.stack([img_w, img_h, img_w, img_h])
            boxes = boxes * scale_fct
            boxes[:,:2] -= boxes[:,2:] / 2
            boxes[:,0::2].clamp_(min=torch.tensor(0, device=images.device), max=img_w)
            boxes[:,1::2].clamp_(min=torch.tensor(0, device=images.device), max=img_h)

            #save results
            
            for i, (box, label) in enumerate(zip(boxes, pred_phrases)):
                if with_logits:
                    category = label[0]
                    score = label[1]
                else:
                    category = label
                    score = 1.0
                if category == '.':
                    continue
                json_dict = {
                    "image_id": targets[j]['image_id'],
                    "category_id": int(img_cat_dic[category]),
                    "bbox": box.tolist(),
                    "score": score,
                    "track_id": i,
                    "video_id": targets[j]['video_id'][0],
                    "is_new": 'true'
                }
                res_json_list.append(json_dict)

        # # print("boxes_filt:", boxes_filt, "pred_phrases:", pred_phrases)
        # if (i+1) % 30 == 0:
        #     used_time = time.time() - start
        #     eta = len(data_loader) / (i+1e-5) * used_time - used_time
        #     print(f"processed {i}/{len(data_loader)} images. time: {used_time:.2f}s, ETA: {eta:.2f}s")

    result_name = 'infer_result' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.json'
    json_path = osp.join(args.output_json, result_name)
    json.dump(res_json_list, open(json_path, 'w'), indent=4)