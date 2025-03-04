# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import time

import torch
import util.misc as utils

from datasets.data_prefetcher import data_dict_to_cuda
from torch.utils.tensorboard import SummaryWriter

step = 0

def train_one_epoch_mot(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, dataset_file: str,
                    max_norm: float = 0, writer: SummaryWriter = None,
                    print_freq: int = 10):
    global step
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        # if (sum(p.numel() for p in data_dict['proposals']) == 0 or data_dict['proposals'] is None) and 'dance' in dataset_file:
        #     print('warning: no proposals in this batch, skip it')
        #     continue
        if not any([len(i) for i in data_dict['gt_instances']]):
            print('warning: no gt instances in this batch, skip it')
            continue
        data_dict = data_dict_to_cuda(data_dict, device)
        cat_names_dict = data_dict['cat_names']
        captions_dict = build_captions_and_token_span(cat_names_dict)
        # start_time = time.time()
        outputs = model(data_dict, captions_dict)

        loss_dict = criterion(outputs, data_dict)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # time1 = time.time()
        extra_loss = outputs['extra_loss']
        losses += extra_loss * 0
        # time2 = time.time()
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        # time3 = time.time()
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        assert torch.isnan(grad_total_norm) == False, "grad is nan"
        optimizer.step()
        # end_time = time.time()
        # print('time0:{}, time1:{}, time2:{}, time3:{}'.format(time1-start_time, time2-time1, time3-time2, end_time-time3))
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        if writer is not None and int(os.environ.get("RANK", 0))==0:
            writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], step)
            # writer.add_scalar('train/lr_backbone', optimizer.param_groups[1]["lr"], step)
            # writer.add_scalar('train/lr_linear_proj', optimizer.param_groups[2]["lr"], step)
            writer.add_scalar('train/loss', loss_value, step)
            writer.add_scalar('train/grad_norm', grad_total_norm, step)
            writer.add_scalars('train/other_losses', loss_dict_reduced_scaled, step)
            step += 1
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def build_captions_and_token_span(cat_dict):

    cat2tokenspan = {}
    captions = ""
    cat_list = list(cat_dict.keys())
    captions_dict = {}
    cap2cat = {}
    
    for catname in cat_list:
        catname_l = catname.lower()

        tokens_positive_i = []
        subnamelist = [i.strip() for i in catname.split("(")[0].replace("_", " ").split(" ")]
        cap = ' '.join(subnamelist).strip()
        cap2cat[cap] = catname
        for subname in subnamelist:
            if len(subname) == 0:
                continue
            if len(captions) > 0:
                captions = captions + " "
            strat_idx = len(captions)
            end_idx = strat_idx + len(subname)
            tokens_positive_i.append([strat_idx, end_idx])
            captions = captions + subname

        if len(tokens_positive_i) > 0:
            captions = captions + " ."
            cat2tokenspan[catname_l] = tokens_positive_i

    captions_dict['captions'] = captions
    captions_dict['cat2tokenspan'] = cat2tokenspan
    captions_dict['cap2cat'] = cap2cat

    return captions_dict