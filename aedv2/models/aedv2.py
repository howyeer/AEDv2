import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import List

from util import box_ops, checkpoint
from .deformable_detr import SetCriterion, focal_loss
from .loss import MultiPosCrossEntropyLoss, L2Loss
from aedv2.models.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid,
                       linear_assignment)

from util.box_ops import box_cxcywh_to_xyxy, box_iou

from .attention import WeightAttention
from .query_buffer import QueryBuffer
from .query_updating import build as build_query_updating_layer

from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
from groundingdino.models.GroundingDINO.transformer import build_transformer
from groundingdino.models.GroundingDINO.backbone import build_backbone
from groundingdino.models.registry import MODULE_BUILD_FUNCS
from groundingdino.util.misc import (
    NestedTensor,
    inverse_sigmoid,
    nested_tensor_from_tensor_list,
)
from groundingdino.util.vl_utils import create_positive_map_from_span
from groundingdino.models.GroundingDINO.bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToPILImage
import cv2
import time

class ClipMatcher(SetCriterion):

    def __init__(self, num_classes,
                        matcher,
                        weight_dict,
                        losses,
                        args):
        super().__init__(num_classes, matcher, weight_dict, losses)
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        self.num_classes = num_classes
        self.matcher = matcher  # abandoned
        self.weight_dict = weight_dict
        self.losses = losses
        self.match_thresh = args.train_match_thresh
        self.train_iou_thresh = args.train_iou_thresh
        self.focal_loss = True
        self.losses_dict = {}
        self._current_frame_idx = 0

        self.mpce_loss = MultiPosCrossEntropyLoss()

    def initialize_for_single_clip(self, gt_instances: List[Instances]):
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}
        self.num_frames = len(gt_instances)

    def _step(self):
        self._current_frame_idx += 1

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float32, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def get_loss(self, loss: str, outputs: dict):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'pt_weights': self.loss_pt_weights,
            'pp_weights': self.loss_pp_weights,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs)

    def loss_boxes(self, outputs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        gt_instances = outputs['gt_instances']
        indices = outputs['indices']
        num_boxes = outputs['num_boxes']
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)

        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0) # size(16)
        mask = (target_obj_ids != -1)

        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[mask]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[mask])))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses
    
    def loss_pt_weights(self, outputs):
        weights_cos = outputs['pt_weights_cos']
        weights_mm = outputs['pt_weights_mm']
        if weights_cos.numel() > 0:
            gt = torch.zeros_like(weights_cos)
            pt_matched_indices = outputs['pt_matched_indices']  # shape: [num_matched_proposals, num_tack_queries]
            gt[pt_matched_indices[:, 0], pt_matched_indices[:, 1]] = 1
            loss_1 = focal_loss(weights_cos,
                            gt,
                            alpha=1-1/weights_cos.shape[1],
                            gamma=2,
                            num_boxes=weights_cos.shape[0], mean_in_dim0=False)
            loss_2 = self.mpce_loss(weights_mm, gt, (gt.sum(dim=1) > 0).float())
            loss = loss_1 + loss_2
        else:
            loss = torch.tensor(0.0).to(self.sample_device)
        # if loss weight is nan
        # if torch.isnan(loss):
        #     raise ValueError("loss weight is nan")
        losses = {'loss_pt_weight': loss}
        return losses

    def loss_pp_weights(self, outputs):
        weights_cos = outputs['pp_weights_cos']
        weights_mm = outputs['pp_weights_mm']
        if weights_cos.numel() > 0:
            gt = torch.eye(weights_cos.shape[1], dtype=torch.float32, device=weights_cos.device)
            loss1 = focal_loss(weights_cos,
                              gt,
                              alpha=1-1/weights_cos.shape[1],
                              gamma=2,
                              num_boxes=weights_cos.shape[0], mean_in_dim0=False)
            loss2 = self.mpce_loss(weights_mm, gt, (gt.sum(dim=1) > 0).float())
            loss = loss1 + loss2
        else:
            loss = torch.tensor(0.0).to(self.sample_device)
        # if loss weight is nan
        # if torch.isnan(loss):
        #     raise ValueError("loss weight is nan")
        losses = {'loss_pp_weight': loss}
        return losses
    
    def loss_cross_clip(self, weights_cos, weights_mm, gt):
        loss = torch.tensor(0.0).to(self.sample_device)
        if weights_cos.numel() > 0:
            loss1 = focal_loss(weights_cos,
                              gt,
                              alpha=1-1/weights_cos.shape[1],
                              gamma=2,
                              num_boxes=weights_cos.shape[0], mean_in_dim0=False)
            loss2 = self.mpce_loss(weights_mm, gt, (gt.sum(dim=1) > 0).float())
            loss = loss1 + loss2
        else:
            loss = torch.tensor(0.0).to(weights_cos)
        self.losses_dict.update({'weight_loss_cross_clip': loss})
    
    def match_for_single_frame(self, outputs: dict, num_proposals: int):
        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
        track_instances: Instances = outputs['track_instances']
        track_instances.matched_gt_ids[:] = -1
        pt_weights_cos = outputs['pt_weights_cos'][0]
        pt_weights_mm = outputs['pt_weights_mm'][0]

        # filter out the extra dets (gt_ids == -2) in proposals
        is_valid = track_instances.gt_ids[:num_proposals] >= -1
        valid_proposal_instances = track_instances[:num_proposals][is_valid]
        invalid_proposal_instances = track_instances[:num_proposals][~is_valid]

        track_query_instances = track_instances[num_proposals:]
        proposal_instances = track_instances[:num_proposals]

        # step1. match proposals and gts
        i, j = torch.where(valid_proposal_instances.gt_ids[:, None] == gt_instances_i.obj_ids)
        matched_indices = torch.stack([i, j], dim=1).to(pt_weights_cos.device)
        valid_proposal_instances.matched_gt_ids[i] = j

        # step2. inherit id and calculate iou
        valid_proposal_instances.obj_ids[matched_indices[:, 0]] = gt_instances_i.obj_ids[matched_indices[:, 1]].long()
        valid_proposal_instances.matched_gt_ids[matched_indices[:, 0]] = matched_indices[:, 1]
        assert torch.all(valid_proposal_instances.obj_ids >= 0) and torch.all(valid_proposal_instances.matched_gt_ids >= 0), \
        "matched gt ids should be >= 0, get {} and {}".format(valid_proposal_instances.obj_ids, valid_proposal_instances.matched_gt_ids)
        active_idxes = (valid_proposal_instances.obj_ids >= 0) & (valid_proposal_instances.matched_gt_ids >= 0)
        active_track_boxes = valid_proposal_instances.pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[valid_proposal_instances.matched_gt_ids[active_idxes]]
            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            valid_proposal_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

        # step3. remove valid_proposal_instances with iou < 0.5 (ambiguous) & update valid
        neg_valid_proposal_mask = valid_proposal_instances.iou < 0.5
        neg_is_valid_idx = torch.where(is_valid)[0][neg_valid_proposal_mask]
        is_valid_mask = torch.ones_like(is_valid, dtype=torch.bool)  # is_valid and iou < 0.5, shape: [num_proposals]
        is_valid_mask[neg_is_valid_idx] = False
        new_is_valid = is_valid[is_valid_mask]
        new_valid_proposal_instances = valid_proposal_instances[~neg_valid_proposal_mask]
        pt_weights_cos, pt_weights_mm = pt_weights_cos[is_valid_mask], pt_weights_mm[is_valid_mask]
        valid_pt_weights_cos = pt_weights_cos[new_is_valid]  # shape: [num_valid_proposals, num_track_queries]
        valid_pt_weights_mm = pt_weights_mm[new_is_valid]  # shape: [num_valid_proposals, num_track_queries]

        # step4. match proposals and track queries
        i, j = torch.where(new_valid_proposal_instances.gt_ids[:, None] == track_query_instances.gt_ids)
        pt_matched_indices = torch.stack([i, j], dim=1).to(pt_weights_cos.device)

        with torch.no_grad():
            for (valid_p_idx, t_idx) in pt_matched_indices:  # matching for valid proposals (not newly apprear)
                similarity = valid_pt_weights_cos[valid_p_idx, t_idx]
                if similarity < 1 - self.match_thresh:
                    new_valid_proposal_instances.obj_ids[valid_p_idx] = -1
                    new_valid_proposal_instances.gt_ids[valid_p_idx] = -2
                else:
                    track_query_instances.obj_ids[t_idx] = -1
                    new_valid_proposal_instances.matched_track_embedding[valid_p_idx] = track_query_instances.query_pos[t_idx]
                    new_valid_proposal_instances.new[valid_p_idx] = False
            
            pt_weight_np = pt_weights_cos.detach().clone().cpu().numpy().astype('float32')
            matches, unmatched_ps, unmatched_ts = linear_assignment(1-pt_weight_np, thresh=self.match_thresh)
            for invalid_p_idx in torch.where(~new_is_valid)[0].cpu().numpy():  # matching for invalid proposals
                assert invalid_p_idx not in pt_matched_indices.cpu().numpy()[:, 0]
                p_idx = (~new_is_valid)[:invalid_p_idx].sum()  # idx in invalid_proposal_instances, different from invalid_p_idx
                if invalid_p_idx in matches[:, 0]:  # matched invalid proposals
                    t_idx = matches[matches[:, 0] == invalid_p_idx][0][1]
                    assert new_is_valid[invalid_p_idx] == False
                    if track_query_instances.gt_ids[t_idx] >= 0:  # mistakenly matched with a valid track query
                        invalid_proposal_instances.gt_ids[p_idx] = -3
                    else:  # matched successfully
                        track_query_instances.gt_ids[t_idx] = -3
                        invalid_proposal_instances.matched_track_embedding[p_idx] = track_query_instances.query_pos[t_idx]
                        invalid_proposal_instances.new[p_idx] = False
                else:  # unmatched invalid proposals
                    invalid_proposal_instances.new[p_idx] = True

        # step5. calculate losses.
        num_boxes = max(len(valid_proposal_instances.pred_boxes), 1)
        outputs_last = {
                        'pred_boxes': valid_proposal_instances.pred_boxes.unsqueeze(0),
                        'indices': [(matched_indices[:, 0], matched_indices[:, 1])],
                        'gt_instances': [gt_instances_i],
                        'pt_weights_cos': valid_pt_weights_cos,
                        'pt_weights_mm': valid_pt_weights_mm,
                        'pt_matched_indices': pt_matched_indices,
                        'pp_weights_cos': outputs['pp_weights_cos'][0],
                        'pp_weights_mm': outputs['pp_weights_mm'][0],
                        'num_boxes': num_boxes,
                        }
        
        self.num_samples += len(gt_instances_i)   # + num_disappear_track
        self.sample_device = pt_weights_cos.device
        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_last)
            self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                outputs_layer = {
                    'pred_boxes': aux_outputs['pred_boxes'][0, is_valid].unsqueeze(0),
                    'indices': [(matched_indices[:, 0], matched_indices[:, 1])],
                    'gt_instances': [gt_instances_i],
                    'pt_weights_cos': aux_outputs['pt_weights_cos'][0, is_valid_mask][new_is_valid],
                    'pt_weights_mm': aux_outputs['pt_weights_mm'][0, is_valid_mask][new_is_valid],
                    'pt_matched_indices': pt_matched_indices,
                    'pp_weights_cos': aux_outputs['pp_weights_cos'][0],
                    'pp_weights_mm': aux_outputs['pp_weights_mm'][0],
                    'num_boxes': num_boxes,
                }
                
                for loss in self.losses:
                    l_dict = self.get_loss(loss,
                                           outputs=outputs_layer)
                    self.losses_dict.update(
                        {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                         l_dict.items()})
        self._step()
        return Instances.cat([new_valid_proposal_instances, invalid_proposal_instances, track_query_instances])

    def forward(self, outputs, input_data: dict):
        # losses of each frame are calculated during the model's forwarding 
        # and are outputted by the model as outputs['losses_dict].
        losses = outputs.pop("losses_dict")
        # num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in losses.items():
            losses[loss_name] /= self.num_frames
        return losses
    
    def _match_det_gt(self, det_boxes, det_scores, det_features, det_labels, gt):
        if len(gt) != 0:
            gt_boxes = gt.boxes
            gt_labels = gt.labels
            gt_obj_ids = gt.obj_ids
            # assert gt_boxes.shape[0] != 0
            iou, union = box_iou(box_ops.box_cxcywh_to_xyxy(det_boxes), box_ops.box_cxcywh_to_xyxy(gt_boxes))
            iou = iou.detach().cpu()
            matches, det_unmached, _ = linear_assignment((1.0-iou).numpy(), thresh=1-self.train_iou_thresh)
            matched_boxes = det_boxes[matches[:, 0]]
            matched_scores = det_scores[matches[:, 0]].unsqueeze(1)
            matched_gt_labels = gt_labels[matches[:, 1]].unsqueeze(1)
            matched_det_labels = det_labels[matches[:, 0]].unsqueeze(1)
            matched_obj_ids = gt_obj_ids[matches[:, 1]].unsqueeze(1)
            matched_features = det_features[matches[:, 0]]
            matched_proposals = torch.cat([matched_boxes, matched_scores, matched_gt_labels, matched_det_labels, matched_obj_ids, matched_features], dim=1)
            unmached_boxes = det_boxes[det_unmached]
            unmached_scores = det_scores[det_unmached].unsqueeze(1)
            unmached_gt_labels = torch.ones_like(unmached_scores) * (-1)
            unmached_det_labels = det_labels[det_unmached].unsqueeze(1)
            unmached_obj_ids = torch.ones_like(unmached_scores) * (-2)
            unmached_features = det_features[det_unmached]
            unmached_proposals = torch.cat([unmached_boxes, unmached_scores, unmached_gt_labels, unmached_det_labels, unmached_obj_ids, unmached_features], dim=1)
            proposals = torch.cat([matched_proposals, unmached_proposals], dim=0)  # [:4,      4,       5,          6,        7,      8:263]
                                                                                   # [boxes, scores, gt_labels, det_labels, obj_ids, features]
        else:
            #-------------------------------------------------------------------------------
            matches = []
            unmached_scores = det_scores.unsqueeze(1)
            unmached_gt_labels = torch.ones_like(unmached_scores) * (-1)
            unmached_obj_ids = torch.ones_like(unmached_scores) * (-2)
            proposals = torch.cat([det_boxes, unmached_scores, unmached_gt_labels, det_labels.unsqueeze(1), 
                                   unmached_obj_ids, det_features], dim=1)
            det_unmached = torch.arange(len(det_boxes))
        # assert matched_boxes.shape == gt_boxes.shape
        return proposals, det_unmached , matches

class RuntimeTrackerBase(object):
    def __init__(self, val_match_high_thresh, val_match_low_thresh, miss_tolerance, match_high_score):
        self.val_match_high_thresh = val_match_high_thresh
        self.val_match_low_thresh = val_match_low_thresh
        self.miss_tolerance = miss_tolerance
        self.match_high_score = match_high_score
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances, weights):  # two stage
        # weight: [num_proposals, track_queries]
        num_proposals = weights.shape[0]
        proposal_instances = track_instances[:num_proposals]
        # split high and low proposals by det score
        high_score_mask = proposal_instances.det_scores > self.match_high_score
        low_score_mask = ~high_score_mask
        high_proposal_instances = proposal_instances[high_score_mask]
        low_proposal_instances = proposal_instances[low_score_mask]
        # get track queries
        track_query_instances = track_instances[num_proposals:]
        device = proposal_instances.obj_ids.device
        assert torch.all(proposal_instances.disappear_time==0)

        # get high score proposals-track queries weights
        high_weights = weights[high_score_mask].cpu().numpy().astype('float32')
        # matching
        high_matches, high_unmatched_p, high_unmatched_t = linear_assignment(1-high_weights, thresh=self.val_match_high_thresh)
        # update
        high_proposal_instances.obj_ids[high_matches[:, 0]] = track_query_instances.obj_ids[high_matches[:, 1]]
        high_proposal_instances.new[high_matches[:, 0]] = False
        high_proposal_instances.matched_track_embedding[high_matches[:, 0]] = track_query_instances.query_pos[high_matches[:, 1]]
        # delet the matched track queries
        track_query_instances.obj_ids[high_matches[:, 1]] = -1

        # get low score proposals-track queries weights
        low_weights = weights[low_score_mask][:, high_unmatched_t].cpu().numpy().astype('float32')
        # matching
        low_matches, low_unmatched_p, low_unmatched_t = linear_assignment(1-low_weights, thresh=self.val_match_low_thresh)
        # update
        low_proposal_instances.obj_ids[low_matches[:, 0]] = track_query_instances.obj_ids[high_unmatched_t[low_matches[:, 1]]]
        low_proposal_instances.new[low_matches[:, 0]] = False
        low_proposal_instances.matched_track_embedding[low_matches[:, 0]] = track_query_instances.query_pos[high_unmatched_t[low_matches[:, 1]]]
        # delet the matched track queries
        track_query_instances.obj_ids[high_unmatched_t[low_matches[:, 1]]] = -1
        
        # assign id for hight new proposals
        num_new_objs = high_unmatched_p.size
        high_proposal_instances.obj_ids[high_unmatched_p] = self.max_obj_id + torch.arange(num_new_objs, device=device)
        self.max_obj_id += num_new_objs

        # delete low new proposals
        low_proposal_instances.obj_ids[low_unmatched_p] = -1  # init is -1, so this is not necessary?

        # update disappear time
        track_query_instances.disappear_time[high_unmatched_t[low_unmatched_t]] += 1

        # delete
        to_del = track_query_instances.disappear_time >= self.miss_tolerance
        track_query_instances.obj_ids[to_del] = -1

        return Instances.cat([high_proposal_instances, low_proposal_instances, track_query_instances])


class TrackerPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, track_instances: Instances, target_size) -> Instances:
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_bbox = track_instances.pred_boxes

        if len(out_bbox) != 0:
            # convert to [x0, y0, x1, y1] format
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            # clip boxes
            boxes = boxes.clamp(min=0, max=1)
            # and from relative [0, 1] to absolute [0, height] coordinates
            img_h, img_w = target_size
            scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
            boxes = boxes * scale_fct[None, :]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32, device=out_bbox.device)

        track_instances.boxes = boxes
        return track_instances  

class AEDv2(GroundingDINO):
    def __init__(
        self, 
        backbone,
        transformer,
        num_queries,
        criterion,
        track_embed,
        num_classes,
        aux_loss=False,
        buffer=None,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
        use_checkpoint=False,
        dropout=0.1,
        box_threshold= 0.3
        ):
        super().__init__(
        backbone,
        transformer,
        num_queries,
        aux_loss,
        iter_update,
        query_dim,
        num_feature_levels,
        nheads,
        # two stage
        two_stage_type,
        dec_pred_bbox_embed_share,
        two_stage_class_embed_share,
        two_stage_bbox_embed_share,
        num_patterns,
        dn_number,
        dn_box_noise_scale,
        dn_label_noise_ratio,
        dn_labelbook_size,
        text_encoder_type,
        sub_sentence_present,
        max_text_len
        )
        self.num_clip = 0
        self.box_threshold = box_threshold

        hidden_dim = self.transformer.d_model
        self.d_model = hidden_dim
        self.post_process = TrackerPostProcess()
        
        _feature_embed = MLP(hidden_dim, hidden_dim*2, hidden_dim, 3)
        for layer in _feature_embed.layers:
            nn.init.normal_(layer.weight, mean=0.0, std=0.1)
            nn.init.constant_(layer.bias, 0)

        feature_embed_layerlist = [_feature_embed for i in range(self.transformer.num_decoder_layers)]  
        self.feature_embed = nn.ModuleList(feature_embed_layerlist)
        self.transformer.decoder.feature_embed = self.feature_embed
        self.requires_set = True

        self.criterion = criterion
        self.track_embed = track_embed
        self.num_classes = num_classes # not used in AED
        self.buffer = buffer
        self.track_base = None
        self.weight_attn = WeightAttention(hidden_dim, 2, attn_drop=dropout)

        # self.det_embed = nn.Embedding(1, hidden_dim)
        self.extra_linear = nn.Linear(hidden_dim, 1)
        self._grad_set()    

    #-------------------------------
    def _grad_set(self):
        for name, param in self.named_parameters():
            if "decoder.bbox_embed" in name or "feature_embed" in name or "weight_attn" in name or "extra_linear" in name:   #"decoder.bbox_embed" in name or
                param.requires_grad_(self.requires_set)
            else:
                param.requires_grad_(False)

    def clear(self):
        if not self.training:
            self.track_base.clear()

    def _generate_empty_tracks(self, proposals=None):
        track_instances = Instances((1, 1))
        _, d_model = self.extra_linear.weight.shape
        if proposals is None:
            track_instances.ref_pts = torch.empty((0, 4), dtype=torch.float32)
            track_instances.pred_boxes = torch.empty((0, 4), dtype=torch.float32)
            track_instances.query_pos = torch.empty((0, d_model), dtype=torch.float32)  # query
            track_instances.labels = torch.empty((0), dtype=torch.long)
            track_instances.det_scores = torch.empty((0), dtype=torch.float32)
        else:
            track_instances.ref_pts = proposals[:, :4]  # [xc, yc, w, h]
            track_instances.pred_boxes = proposals[:, :4]  # [xc, yc, w, h]
            track_instances.query_pos = pos2posemb(proposals[:, 4:5], d_model) + proposals[:, 8:]  # query
            track_instances.labels = proposals[:,6].long()
            track_instances.det_scores = proposals[:,4].float()
        track_instances.output_embedding = torch.zeros((len(track_instances), d_model))
        track_instances.obj_ids = torch.full((len(track_instances),), -1, dtype=torch.long)
        track_instances.matched_gt_ids = torch.full((len(track_instances),), -1, dtype=torch.long)
        track_instances.disappear_time = torch.zeros((len(track_instances), ), dtype=torch.long)
        track_instances.iou = torch.zeros((len(track_instances),), dtype=torch.float32)
        track_instances.save_period = torch.zeros((len(track_instances), ), dtype=torch.float32)
        track_instances.matched_track_embedding = torch.zeros((len(track_instances), d_model), dtype=torch.float32)
        
        track_instances.new = torch.ones((len(track_instances),), dtype=torch.bool)  # bool is new or not
        if self.training:
            if proposals is None:
                track_instances.gt_ids = torch.empty((0), dtype=torch.long)
            else:
                # gt_ids = -3: invalid extra dets (need to be removed)
                # gt_ids = -2: valid extra dets (e.g. false positives or unlabeled dets)
                # gt_ids = -1: untracked
                # gt_ids >= 0: tracked
                track_instances.gt_ids = proposals[:,7].long()
        return track_instances.to(self.extra_linear.weight.device)

    def _forward_single_image(self, samples: NestedTensor, targets: List = None, **kw):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if targets is None:
            captions = kw["captions"]
        else:
            captions = [t["caption"] for t in targets]

        # encoder texts
        tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(
            samples.device
        )
        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            # import ipdb; ipdb.set_trace()
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195
        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]

        text_dict = {
            "encoded_text": encoded_text,  # bs, text_len, d_model
            "text_token_mask": text_token_mask,  # bs, text_len
            "position_ids": position_ids,  # bs, text_len
            "text_self_attention_masks": text_self_attention_masks,  # bs, text_len,text_len
        }

        # import ipdb; ipdb.set_trace()
        # if isinstance(samples, (list, torch.Tensor)):
        #     samples = nested_tensor_from_tensor_list(samples)
        # if not hasattr(self, 'features') or not hasattr(self, 'poss'):
        #     self.set_image_tensor(samples)
        # inf_start_time = time.time()
        features, poss = self.backbone(samples)
        
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)
        # time_backbone = time.time() 
        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
        )
        # time_transformer = time.time()
        # print('backbone time: {} transformer time: {}'.format(time_backbone - inf_start_time, time_transformer - time_backbone))
        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # output
        outputs_logit = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )
        # feature                 256
        outputs_feature = torch.stack(
            [
                layer_feature_embed(layer_hs) 
                for layer_feature_embed, layer_hs in zip(self.feature_embed, hs)
            ]
        )
        # # for intermediate outputs
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)

        # # for encoder output
        # if hs_enc is not None:
        #     # prepare intermediate outputs
        #     interm_coord = ref_enc[-1]
        #     interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
        #     out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
        #     out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}
        # unset_image_tensor = kw.get('unset_image_tensor', True)
        # if unset_image_tensor:
        #     self.unset_image_tensor() ## If necessary

        output_all_layer = {"pred_logits": outputs_logit, "pred_boxes": outputs_coord_list, "pred_features": outputs_feature}

        return output_all_layer
    
    def get_groundingdino_output(self, caption, logits, boxes, features, text_threshold=None, with_logits=True, token_spans=None):
        assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
        positive_maps = create_positive_map_from_span(
                self.tokenizer(caption),
                token_span=token_spans
            ).to(logits.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        all_features = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > self.box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            # filt features
            all_features.append(features[filt_mask])
            all_phrases.extend([phrase for _ in range(len(logit_phr[filt_mask]))])
        boxes_filt = torch.cat(all_boxes, dim=0)
        logits_filt = torch.cat(all_logits, dim=0)
        pred_phrases = all_phrases
        features_filt = torch.cat(all_features, dim=0)
        return boxes_filt, logits_filt, features_filt, pred_phrases
    
    def _calculate_weight_matrix(self, proposals_embed, track_queries_embed, attn_mask=None):  # p_query_pos, t_query_pos
        # get proposal-track attention weights
        pt_weights_cos, pt_weights_mm = self.weight_attn(proposals_embed, track_queries_embed, None)
        # get proposal-proposal attention weights
        pp_weights_cos, pp_weights_mm = self.weight_attn(proposals_embed, proposals_embed, None)

        out = {'pt_weights_cos': pt_weights_cos, 'pp_weights_cos': pp_weights_cos,
               'pt_weights_mm': pt_weights_mm, 'pp_weights_mm': pp_weights_mm}
        out['proposals_embed'] = proposals_embed
        return out
    
    def _post_process_single_image(self, frame_res, track_instances, num_proposals, is_last):        
        track_instances.output_embedding[:num_proposals] = frame_res['proposals_embed']      #[0, :num_proposals]
        # matched_track_embedding inits as its own embedding
        track_instances.matched_track_embedding[:num_proposals] = frame_res['proposals_embed']    #[0, :num_proposals]
        
        if self.training:
            # the track id will be assigned by the mather.
            frame_res['track_instances'] = track_instances
            # only proposal_instances are keeped
            track_instances = self.criterion.match_for_single_frame(frame_res, num_proposals)
            # track_instances = track_instances[:num_proposals]
            if self.buffer is not None:
                self.buffer(track_instances[:num_proposals])
                if is_last:
                    weights_ids = []
                    embeddings = torch.empty((0, self.extra_linear.weight.shape[1]), dtype=torch.float32, device=track_instances.query_pos.device)
                    ids = torch.empty((0, ), dtype=torch.long, device=track_instances.query_pos.device)
                    for id, tmp in self.buffer.memory.items():
                        embeddings = torch.cat([embeddings, tmp['embeddings']], dim=0)
                        ids = torch.cat([ids, torch.full((tmp['embeddings'].shape[0],), id, dtype=torch.long, device=track_instances.query_pos.device)], dim=0)
                    q = embeddings.clone()
                    k = embeddings.clone()
                    gt = (ids.unsqueeze(0) == ids.unsqueeze(1)).float().to(q.device)
                    weights_cos, weights_mm = self.weight_attn(q[None, :], k[None, :])
                    self.criterion.loss_cross_clip(weights_cos[0], weights_mm[0], gt)
        else:
            # each track will be assigned an unique global id by the track base.
            pt_weights = frame_res['pt_weights_cos']
            track_instances = self.track_base.update(track_instances, pt_weights[0])
            
            # for weight visualization
            track_instances.gt_ids = track_instances.obj_ids.clone()
            if self.buffer is not None:
                self.buffer(track_instances[:num_proposals])
                if is_last:
                    weights_ids = {}
                    embeddings = torch.empty((0, self.extra_linear.weight.shape[1]), dtype=torch.float32, device=track_instances.query_pos.device)
                    for id, tmp in self.buffer.memory.items():
                        embeddings = torch.cat([embeddings, tmp['embeddings']], dim=0)
                    total_q = embeddings.clone()
                    total_k = embeddings.clone()
                    weights_cos, _ = self.weight_attn(total_q[None, :], total_k[None, :])
                    weights_ids['total'] = weights_cos[0]
                    frame_res['cross_clip_weight'] = weights_ids

        tmp = {}
        tmp['track_instances'] = track_instances
        out_track_instances, num_active_proposals, active_idxes, active_proposals = self.track_embed(tmp, num_proposals)
        frame_res['track_instances'] = out_track_instances
        frame_res['active_idxes'] = active_idxes

        return frame_res, active_proposals
    
    @torch.no_grad()
    def inference_single_image(self, img, ori_img_size, track_instances=None, captions_dict=None, is_last=False):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        frame_res_all_layer = self._forward_single_image(img, captions=[captions_dict['captions']])
        # 从900个候选中筛选出大于box_threshold的框
        frame_res_logits = frame_res_all_layer['pred_logits'][-1].sigmoid()[0]
        frame_res_boxes = frame_res_all_layer['pred_boxes'][-1][0]
        frame_res_features = frame_res_all_layer['pred_features'][-1][0]
        boxes_filt, logits_filt, features_filt, pred_phrases = self.get_groundingdino_output(captions_dict['captions'], frame_res_logits,
                                                                                frame_res_boxes, frame_res_features,
                                                                                token_spans=captions_dict['cat2tokenspan'].values())
        labels_filt = []
        for pred_phrase in pred_phrases:
            catname = captions_dict['cap2cat'][pred_phrase]
            labels_filt.append(captions_dict['cat_names'][catname])
        labels_filt = torch.tensor(labels_filt).to(boxes_filt.device)

        proposals_scores = logits_filt.unsqueeze(1)
        proposals_gt_labels = torch.ones_like(proposals_scores) * (-1)
        proposals_obj_ids = torch.ones_like(proposals_scores) * (-2)
        proposals = torch.cat([boxes_filt, proposals_scores, proposals_gt_labels, labels_filt.unsqueeze(1), 
                                proposals_obj_ids, features_filt], dim=1)
        
        num_proposals = len(proposals) if proposals is not None else 0
        if track_instances is None:
            track_instances = self._generate_empty_tracks(proposals)
        else:
            track_instances = Instances.cat([
                self._generate_empty_tracks(proposals),
                track_instances])
        
        query_embed = track_instances.query_pos
        tgt = query_embed.unsqueeze(0)
        proposals_embed = tgt[:, :num_proposals]
        track_queries_embed = tgt[:, num_proposals:]
        attn_mask = None

        #计算特征相似度矩阵
        res = self._calculate_weight_matrix(proposals_embed, track_queries_embed)
        res['pred_boxes'] = proposals[:, :4] if proposals is not None else torch.empty((0, 4))

        res, active_proposals = self._post_process_single_image(res, track_instances, int(num_proposals), is_last)
        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size)
        ret = {'track_instances': track_instances, 'num_active_proposals': active_proposals.sum(), 'res': res}
        if 'ref_pts' in res:
            ref_pts = res['ref_pts']
            img_h, img_w = ori_img_size
            scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            ref_pts = ref_pts * scale_fct[None]
            ret['ref_pts'] = ref_pts
        return ret

    def  forward(self, data:dict, captions_dict):
        self.num_clip += 1
        if self.training:
            self.criterion.initialize_for_single_clip(data['gt_instances'])
        frames = data['imgs']
        outputs = {
            'pred_boxes': [],
        }
        track_instances = None
        proposals = None
        if self.buffer is not None:
            self.buffer.clear()

        for frame_index, (frame, gt, ori_img) in enumerate(zip(frames, data['gt_instances'], data['ori_imgs'])):
            frame.requires_grad = False
            is_last = frame_index == len(frames) - 1
            # star_time = time.time()

            frame = nested_tensor_from_tensor_list([frame])
            frame_res_all_layer = self._forward_single_image(frame, captions=[captions_dict['captions']])
            # time1 = time.time()
            # 从900个候选中筛选出大于box_threshold的框
            frame_res_logits = frame_res_all_layer['pred_logits'][-1].sigmoid()[0]
            frame_res_boxes = frame_res_all_layer['pred_boxes'][-1][0]
            frame_res_features = frame_res_all_layer['pred_features'][-1][0]
            
            #计算额外的损失，确保梯度回传正常
            extra_loss = self.extra_linear(frame_res_features).sum()

            boxes_filt, logits_filt, features_filt, pred_phrases = self.get_groundingdino_output(captions_dict['captions'], frame_res_logits,
                                                                                    frame_res_boxes, frame_res_features,
                                                                                    token_spans=captions_dict['cat2tokenspan'].values())
            # time2 = time.time()

            labels_filt = []
            for pred_phrase in pred_phrases:
                catname = captions_dict['cap2cat'][pred_phrase]
                labels_filt.append(data['cat_names'][catname])
            labels_filt = torch.tensor(labels_filt).to(boxes_filt.device)
            catid2cat = {}
            for id, cat in zip(data['cat_names'].values(), data['cat_names'].keys()):
                catid2cat[id] = cat
            gtcat = [catid2cat[int(catid)] for catid in gt.labels]
            
            #生成proposals
            #与gt框进行iou匹配,存在未匹配的检测框用于后续伪标签处理
            proposals, det_unmached, matches = self.criterion._match_det_gt(boxes_filt, logits_filt, features_filt, labels_filt, gt)

            num_proposals = len(proposals) if proposals is not None else 0
            
            if track_instances is None:
                track_instances = self._generate_empty_tracks(proposals)
            else:
                track_instances = Instances.cat([self._generate_empty_tracks(proposals), track_instances])

            query_embed = track_instances.query_pos
            tgt = query_embed.unsqueeze(0)
            proposals_embed = tgt[:, :num_proposals]
            track_queries_embed = tgt[:, num_proposals:]
            attn_mask = None

            #计算特征相似度矩阵
            frame_res = self._calculate_weight_matrix(proposals_embed, track_queries_embed)
            frame_res['pred_boxes'] = proposals[:, :4] if proposals is not None else torch.empty((0, 4))

            frame_res, active_proposals = self._post_process_single_image(frame_res, track_instances, int(num_proposals), is_last)         
            # time3 = time.time()
            track_instances = frame_res['track_instances']

            #可视化出检测结果
            frame_v = frame.tensors[0]
            predcat = [catid2cat[int(catid)] for catid in proposals[active_proposals, 6]]
            track_ids = track_instances.obj_ids[:active_proposals.sum()].cpu().numpy()

            # self.visualize(frame_v, frame_index, proposals[active_proposals, :4], proposals[active_proposals, 4], predcat, track_ids, 'det')
            # self.visualize(frame_v, frame_index, gt.boxes, gt.labels, gtcat, gt.obj_ids, 'gt')

            outputs['pred_boxes'].append(frame_res['pred_boxes'])

            # print('forward_single_image time:{}s, get_groundingdino_output time:{}s, post_process_single_image time:{}s'\
            #       .format(time1-star_time, time2-time1, time3-time2))
            
        if not self.training:
            outputs['track_instances'] = track_instances
        else:
            outputs['losses_dict'] = self.criterion.losses_dict

        outputs['extra_loss'] = extra_loss
        return outputs

    def visualize(self ,frame, frame_index, boxes_, info1, info2, info3, data_type):
        def plot_bbox(img, bbox, label, track_id):
            tl = 3
            if track_id >= 0:
                np.random.seed(track_id)
            else:
                np.random.seed(2**32 - 1)
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            bbox[:2] -= bbox[2:]/2
            bbox[2:] += bbox[:2]
            bbox *= np.array([w, h, w, h], dtype=np.float32)
            c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(img, c1, c2, color, thickness=tl)
            if label:
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 3
                cv2.rectangle(img, c1, c2, color, -1)  # filled
                cv2.putText(img,
                            label, (c1[0], c1[1] + t_size[1] + 3),
                            0,
                            tl / 4, [225, 255, 255],
                            thickness=tf,
                            lineType=cv2.LINE_AA)
            return img
        img = frame.permute(1, 2, 0).cpu().numpy()
        boxes = boxes_.cpu().detach().numpy()
        # reverse normalize
        img *= np.array([0.229, 0.224, 0.225])
        img += np.array([0.485, 0.456, 0.406])
        img *= 255.0
        h, w, _ = img.shape
        img = cv2.UMat(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if data_type == 'det':
            logits = info1.cpu().detach().numpy()
            phrases = info2
            track_ids = info3
            for phrase, box, logit, track_id in zip(phrases, boxes, logits, track_ids):
                label = str(phrase)+' '+f"{logit:.3f}"
                img = plot_bbox(img, box, label, track_id)
            cv2.imwrite('aedv2/output_dir/output_imgs/det_clip_{}_frame_{}.jpg'.format(self.num_clip, frame_index), img)
        elif data_type == 'gt':
            labels = info2
            obj_ids = info3.cpu().numpy().astype(int)
            for label, box, obj_id in zip(labels, boxes, obj_ids):
                img = plot_bbox(img, box, label, obj_id)
            cv2.imwrite('aedv2/output_dir/output_imgs/gt_clip_{}_frame_{}.jpg'.format(self.num_clip, frame_index), img)



def pos2posemb(pos, num_pos_feats=64, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    posemb = pos[..., None] / dim_t
    posemb = torch.stack((posemb[..., 0::2].sin(), posemb[..., 1::2].cos()), dim=-1).flatten(-3)
    return posemb        


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    





@MODULE_BUILD_FUNCS.registe_with_name(module_name="groundingaed")
def build(args):
    #--------------------------------AED---------------------------------

    num_classes = 1
    device = torch.device(args.device)
    query_updating_layer = build_query_updating_layer(args)

    num_frames_per_batch = max(args.sampler_lengths)
    weight_dict = {}
    for i in range(num_frames_per_batch):
        weight_dict.update({'frame_{}_loss_bbox'.format(i): args.bbox_loss_coef,
                            'frame_{}_loss_giou'.format(i): args.giou_loss_coef,
                            'frame_{}_loss_pt_weight'.format(i): args.pt_weight_loss_coef,
                            'frame_{}_loss_pp_weight'.format(i): args.pp_weight_loss_coef,
                            })

    # TODO this is a hack
    if args.aux_loss:
        for i in range(num_frames_per_batch):
            for j in range(args.dec_layers - 1):
                weight_dict.update({'frame_{}_aux{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_aux{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    'frame_{}_aux{}_loss_pt_weight'.format(i, j): args.pt_weight_loss_coef,
                                    'frame_{}_aux{}_loss_pp_weight'.format(i, j): args.pp_weight_loss_coef,
                                    })
            for j in range(args.dec_layers):
                weight_dict.update({'frame_{}_ps{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_ps{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    'frame_{}_ps{}_loss_pt_weight'.format(i, j): args.pt_weight_loss_coef,
                                    'frame_{}_ps{}_loss_pp_weight'.format(i, j): args.pp_weight_loss_coef,
                                    })
    # buffer = None
    buffer = QueryBuffer()
    if buffer is not None:
        weight_dict.update({'weight_loss_cross_clip': args.cross_clip_weight_loss_coef})
    losses = ['pt_weights', 'pp_weights']
    if args.with_box_refine:
        losses += ['boxes']
    criterion = ClipMatcher(num_classes, matcher=None, weight_dict=weight_dict, losses=losses, args=args)
    criterion.to(device)
    postprocessors = {}

    #---------------------------Grounding-DINO--------------------------------

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    model = AEDv2(
        backbone,
        transformer,
        num_queries=args.num_queries,
        criterion=criterion,
        track_embed=query_updating_layer,
        num_classes=num_classes,
        aux_loss=args.aux_loss,
        buffer=buffer,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
        use_checkpoint=args.use_checkpoint,
        dropout=0.1
    )

    return model, criterion



