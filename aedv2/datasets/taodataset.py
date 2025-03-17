import torch
import copy
import sys
import cv2
import random
import os.path as osp
import numpy as np
import json
import aedv2.datasets.transforms as T

from tao.toolkit.tao import Tao
from torch.utils.data import Dataset
from PIL import Image
from models.structures.instances import Instances
from collections import defaultdict
from torchvision.ops import nms
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from util.misc import linear_assignment
from random import choice, randint


class TAODatasetTrain(Dataset):  # TAO dataset
    def __init__(self, tao_root: str, args, logger, transform=None, one_class=True):
        super().__init__()
        self.root = tao_root
        annotation_path = osp.join(tao_root, 'annotations', 'train_ours_v1.json')
        print('using annotation path {}'.format(annotation_path))
        tao = Tao(annotation_path, logger)
        cats = tao.cats
        vid_img_map = tao.vid_img_map
        img_ann_map = tao.img_ann_map
        vids = tao.vids  # key: video id, value: video info
        self.args = args
        self.num_frames_per_batch = args.sampler_lengths[0]
        self.transform = transform
        self.one_class = one_class  # if true, take all objects as foreground
        self.all_frames_with_gt, self.all_indices, self.vid_tmax, categories_counter = self._generate_train_imgs(vid_img_map, img_ann_map, vids, cats,
                                                                                                                 args.train_base)
        categories_counter = sorted(categories_counter.items(), key=lambda x: x[0])
        print('found {} videos, {} imgs'.format(len(vids), len(self.all_indices)))
        print('number of categories: {}'.format(len(categories_counter)))

        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval


    def _generate_train_imgs(self, vid_img_map, img_ann_map, vids, cats, base_only):
        if base_only:
            print('only use base classes')
        all_frames_with_gt = {}
        all_indices = []
        vid_tmax = {}
        categories_counter = defaultdict(int)
        for vid_info in vids.values():
            vid_id = vid_info['id']
            imgs = vid_img_map[vid_id]
            imgs = sorted(imgs, key=lambda x: x['frame_index'])
            num_imgs = len(imgs)
            cat_name_list = [item['name'] for item in cats.values()]
            cat_dict = dict()
            targets = []  # gt and detection results
            img_infos = []
            cur_vid_indices = []
            for i in range(len(imgs)):
                img = imgs[i]
                cur_vid_indices.append((vid_id, i))
                gt_boxes, gt_labels, gt_track_ids, gt_scores, gt_iscrowd = [], [], [], [], []
                height, width = float(img['height']), float(img['width'])
                anns = img_ann_map[img['id']]
                img_id = img['id']
                img_infos.append({'file_name': img['file_name'],
                                'height': height,
                                'width': width,
                                'frame_index': img['frame_index'],
                                'image_id': img_id,
                                'video_id': vid_id})
                for ann in anns:
                    assert ann['iscrowd'] != 1
                    if base_only and cats[ann['category_id']]['frequency'] == 'r':  # ignore rare classes
                        continue
                    box = ann['bbox']  # x0, y0, w, h
                    box[2] += box[0]
                    box[3] += box[1]
                    gt_boxes.append(box)
                    categories_counter[ann['category_id']] += 1
                    gt_labels.append(ann['category_id'])  # category
                    gt_track_ids.append(ann['track_id'])
                    gt_scores.append(1.0)
                    gt_iscrowd.append(ann['iscrowd'])
                    cat_name = cat_name_list[ann['category_id']-1]
                    cat_dict[cat_name] = ann['category_id']
                if len(gt_track_ids) == 0:
                    targets.append({})
                    continue
                gt_boxes = torch.as_tensor(gt_boxes, dtype=torch.float32)
                gt_labels = torch.as_tensor(gt_labels, dtype=torch.long)
                gt_scores = torch.as_tensor(gt_scores, dtype=torch.float32)
                gt_track_ids = torch.as_tensor(gt_track_ids, dtype=torch.float32)
                gt_iscrowd = torch.as_tensor(gt_iscrowd, dtype=torch.bool)
                targets.append({'boxes': gt_boxes,  # x0, y0, x1, y1
                                'labels': gt_labels,
                                'scores': gt_scores,
                                'obj_ids': gt_track_ids,
                                'iscrowd': gt_iscrowd,
                                })
                akeys = targets[-1].keys()
                if 'boxes' not in akeys:
                    print('---------------------------')
            vid_tmax[vid_id] = i
            cur_vid_indices = [cur_vid_indices[i] for i in range(0, len(cur_vid_indices), self.args.clip_gap)]
            all_indices.extend(cur_vid_indices)
            all_frames_with_gt[vid_id] = {'img_infos': img_infos,
                                          'targets': targets,
                                          'cat_dict': cat_dict,
                                          }
        return all_frames_with_gt, all_indices, vid_tmax, categories_counter
    
    def _targets_to_instances(self, targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = targets['boxes']
        if self.one_class:
            gt_instances.labels = torch.zeros_like(targets['labels'], dtype=targets['labels'].dtype)
        else:
            gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        return gt_instances
    
    def _generate_empty_instance(self, img_shape):
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = torch.empty((0, 4), dtype=torch.float32)
        gt_instances.labels = torch.empty((0,), dtype=torch.int64)
        gt_instances.obj_ids = torch.empty((0,), dtype=torch.float32)
        return gt_instances
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.has_vised = 0
        return
    
    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    def __len__(self):
        return len(self.all_indices)
    
    def sample_indices(self, vid, f_index):
        assert self.sample_mode == 'random_interval'
        tmax = self.vid_tmax[vid]
        ids = [f_index]
        for i in range(1, self.num_frames_per_batch):
            id_ = ids[-1] + randint(1, self.sample_interval)
            while id_ > tmax:
                id_ = id_ - tmax - 1
            ids.append(id_)
        return ids
    
    def __getitem__(self, idx: int):
        vid, f_index = self.all_indices[idx]          # 视频id和帧id
        indices = self.sample_indices(vid, f_index)
        img_infos, targets = [], []
        for i in indices:
            img_infos.append(self.all_frames_with_gt[vid]['img_infos'][i])
            targets.append(self.all_frames_with_gt[vid]['targets'][i])
        cat_names = self.all_frames_with_gt[vid]['cat_dict']
        
        ori_images = [Image.open(osp.join(self.root, 'frames', img_info['file_name'])) \
                  for img_info in img_infos]
        if self.transform is not None:
            images, targets = self.transform(ori_images, targets)
        else:
            raise ValueError('transform is None')
        gt_instances = []
        for img_i, targets_i in zip(images, targets):
            if 'boxes' in targets_i:
            # gt
                gt_instance = self._targets_to_instances(targets_i, img_i.shape[1:3])
                gt_instances.append(gt_instance)
            else:
                # gt
                gt_instances.append(self._generate_empty_instance(img_i.shape[1:3]))
            # det results
        if self.args.shuffle_clip:
            indices = list(range(len(images)))
            random.shuffle(indices)
            images = [images[i] for i in indices]
            img_infos = [img_infos[i] for i in indices]
            gt_instances = [gt_instances[i] for i in indices]
            ori_images = [ori_images[i] for i in indices]
        return {
            'imgs': images,
            'img_infos': img_infos,
            'gt_instances': gt_instances,
            'ori_imgs': ori_images,  
            'cat_names': cat_names
        }
    
class TAODatasetVal(Dataset):  # TAO dataset
    def __init__(self, tao_root: str, args, logger, transform=None, one_class=True):
        super().__init__()
        self.root = tao_root
        if args.split == 'val':
            annotation_path = osp.join(tao_root, 'annotations', 'validation_ours_v1.json')
        else:
            annotation_path = osp.join(tao_root, 'annotations', 'tao_test_burst_v1.json')
        print('using annotation path {}'.format(annotation_path))
        tao = Tao(annotation_path, logger)
        self.tao = tao
        cats = tao.cats
        vid_img_map = tao.vid_img_map
        img_ann_map = tao.img_ann_map
        vids = tao.vids  # key: video id, value: video info
        self.args = args
        self.num_frames_per_batch = args.sampler_lengths[0]
        self.transform = transform
        self.one_class = one_class  # if true, take all objects as foreground
        self.clips_with_gt = self._generate_val_clips(vid_img_map, img_ann_map, vids, cats)
        print('found {} videos, {} clips'.format(len(vids), len(self.clips_with_gt)))

    def _generate_val_clips(self, vid_img_map, img_ann_map, vids, cats):
        clips_with_gt = []
        all_indices = []
        vid_tmax = {}
        categories_counter = defaultdict(int)
        for vid_info in vids.values():
            vid_id = vid_info['id']
            imgs = vid_img_map[vid_id]
            imgs = sorted(imgs, key=lambda x: x['frame_index'])
            num_imgs = len(imgs)
            cat_name_list = [item['name'] for item in cats.values()]
            cat_dict = dict()
            targets = []  # gt and detection results
            img_infos = []
            for i in range(len(imgs)):
                img = imgs[i]
                gt_boxes, gt_labels, gt_track_ids = [], [], []
                height, width = float(img['height']), float(img['width'])
                anns = img_ann_map[img['id']]
                img_id = img['id']
                img_infos.append({'file_name': img['file_name'],
                                'height': height,
                                'width': width,
                                'frame_index': img['frame_index'],
                                'image_id': img_id,
                                'video_id': vid_id})
                for ann in anns:
                    box = ann['bbox']  # x0, y0, w, h
                    box[2] += box[0]
                    box[3] += box[1]
                    gt_boxes.append(box)
                    categories_counter[ann['category_id']] += 1
                    gt_labels.append(ann['category_id'])  # category
                    gt_track_ids.append(ann['track_id'])
                    cat_name = cat_name_list[ann['category_id']-1]
                    cat_dict[cat_name] = ann['category_id']
                if len(gt_track_ids) == 0:
                    targets.append({})
                    continue
                gt_boxes = torch.as_tensor(gt_boxes, dtype=torch.float32)
                gt_labels = torch.as_tensor(gt_labels, dtype=torch.long)
                gt_track_ids = torch.as_tensor(gt_track_ids, dtype=torch.float32)
                targets.append({'boxes': gt_boxes,  # x0, y0, x1, y1
                                'labels': gt_labels,
                                'obj_ids': gt_track_ids,
                                })
            clips_with_gt.append({'img_infos': img_infos,
                                          'targets': targets,
                                          'cat_dict': cat_dict,
                                          })
        return clips_with_gt
    
    def _targets_to_instances(self, targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = targets['boxes']
        if self.one_class:
            gt_instances.labels = torch.zeros_like(targets['labels'], dtype=targets['labels'].dtype)
        else:
            gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        return gt_instances

    def _generate_empty_instance(self, img_shape):
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = torch.empty((0, 4), dtype=torch.float32)
        gt_instances.labels = torch.empty((0,), dtype=torch.int64)
        gt_instances.obj_ids = torch.empty((0,), dtype=torch.float32)
        return gt_instances
      
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        return
    
    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)
    
    def __len__(self):
        return len(self.clips_with_gt)
    
    def __getitem__(self, idx: int):
        gt_instances = []
        img_infos = copy.deepcopy(self.clips_with_gt[idx]['img_infos'])
        targets = copy.deepcopy(self.clips_with_gt[idx]['targets'])
        cat_names = copy.deepcopy(self.clips_with_gt[idx]['cat_dict'])
        
        ori_images = [Image.open(osp.join(self.root, 'frames', img_info['file_name'])) \
                  for img_info in img_infos]
        if self.transform is not None:
            images,targets = self.transform(ori_images, targets)
        else:
            raise ValueError('transform is None')
        for img_i, targets_i in zip(images, targets):
            if 'boxes' in targets_i:
            # gt
                gt_instance = self._targets_to_instances(targets_i, img_i.shape[1:3])
                gt_instances.append(gt_instance)
            else:
                # gt
                gt_instances.append(self._generate_empty_instance(img_i.shape[1:3]))
        return {
            'imgs': images,
            'img_infos': img_infos,
            'gt_instances': gt_instances,
            'ori_imgs': ori_images,  
            'cat_names': cat_names
        }
    
    
def clip_box(box, h, w):
    box[0] = max(0, box[0])
    box[1] = max(0, box[1])
    box[2] = min(w, box[2])
    box[3] = min(h, box[3])
    return box
    
def make_transforms_for_TAO(image_set, args=None):

    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]
    scales = [800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120]

    if image_set == 'train':
        return T.MotCompose([
            T.MotRandomHorizontalFlip(),
            T.MotRandomSelect(
                T.MotRandomResize(scales, max_size=1536),
                T.MotCompose([
                    T.MotRandomResize([1000, 1200]),
                    T.FixedMotRandomCrop(1000, 1200),
                    T.MotRandomResize(scales, max_size=1536),
                ])
            ),
            T.MOTHSV(),
            normalize,
        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_transform(args, image_set):
    train = make_transforms_for_TAO('train', args)
    test = make_transforms_for_TAO('val', args)

    if image_set == 'train':
        return train
    elif image_set == 'val' or image_set == 'exp' or image_set == 'val_gt':
        return test
    else:
        raise NotImplementedError()
    
def build(image_set, args):
    root = osp.join(args.mot_path, 'TAO')
    assert osp.exists(root), 'provided MOT path {} does not exist'.format(root)
    transform = build_transform(args, image_set)
    if image_set == 'train':
        dataset = TAODatasetTrain(root, args, logger=None, transform=transform, one_class=False)
    if image_set == 'val':
        dataset = TAODatasetVal(root, args, logger=None, transform=transform, one_class=False)
    return dataset


#数据集子集
class SubsetDataset(TAODatasetTrain):
    def __init__(self, original_dataset, fraction=0.5):
        self.original_dataset = original_dataset
        self.size = int(fraction * len(original_dataset))
        self.all_indices = original_dataset.all_indices[:self.size]  # 取前 fraction 部分

    def __len__(self):
        return len(self.all_indices)

    def __getitem__(self, idx):
        return self.original_dataset[idx]



if __name__ == '__main__':
    tao = TAODatasetTrain()