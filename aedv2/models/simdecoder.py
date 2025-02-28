# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from models.structures import Boxes, matched_boxlist_iou, pairwise_iou

from util.misc import inverse_sigmoid
from util.box_ops import box_cxcywh_to_xyxy
from ops.modules import MSDeformAttn
from attention import WeightAttention


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300, decoder_self_cross=True, sigmoid_attn=False,
                 extra_track_attn=False, memory_bank=False):
        super().__init__()

        self.new_frame_adaptor = None
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_decoder_layers = num_decoder_layers

        decoder_layer = SimDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, decoder_self_cross,
                                                          sigmoid_attn=sigmoid_attn, extra_track_attn=extra_track_attn,
                                                          memory_bank=memory_bank)
        self.decoder = SimDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))


class ReLUDropout(torch.nn.Dropout):
    def forward(self, input):
        return relu_dropout(input, p=self.p, training=self.training, inplace=self.inplace)

def relu_dropout(x, p=0, inplace=False, training=False):
    if not training or p == 0:
        return x.clamp_(min=0) if inplace else x.clamp(min=0)

    mask = (x < 0) | (torch.rand_like(x) > 1 - p)
    return x.masked_fill_(mask, 0).div_(1 - p) if inplace else x.masked_fill(mask, 0).div(1 - p)


class SimDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, self_cross=True, sigmoid_attn=False,
                 extra_track_attn=False, memory_bank=False):
        super().__init__()

        self.self_cross = self_cross
        self.num_head = n_heads
        self.memory_bank = memory_bank

        # attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.weight_attn = WeightAttention(d_model, 2, attn_drop=dropout)

        # ffn for proposals
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout_relu1 = ReLUDropout(dropout, True)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # memory bank
        if self.memory_bank:
            self.temporal_attn = nn.MultiheadAttention(d_model, 8, dropout=0)
            self.temporal_fc1 = nn.Linear(d_model, d_ffn)
            self.temporal_fc2 = nn.Linear(d_ffn, d_model)
            self.temporal_norm1 = nn.LayerNorm(d_model)
            self.temporal_norm2 = nn.LayerNorm(d_model)

            position = torch.arange(5).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(5, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        # update track query_embed
        self.extra_track_attn = extra_track_attn
        if self.extra_track_attn:
            print('Training with Extra Self Attention in Every Decoder.', flush=True)
            self.update_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout5 = nn.Dropout(dropout)
            self.norm4 = nn.LayerNorm(d_model)

        if self_cross:
            print('Training with Self-Cross Attention.')
        else:
            print('Training with Cross-Self Attention.')

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn_proposal(self, tgt):
        tgt2 = self.linear2(self.dropout_relu1(self.linear1(tgt)))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def _forward_weight_attn(self, query, query_pos, key, key_pos, attn_mask=None):
        q = query
        k = key
        if attn_mask is not None:
            weight_cos, weight_mm = self.weight_attn(q, k, attn_mask=attn_mask)
        else:
            weight_cos, weight_mm = self.weight_attn(q, k)
        return weight_cos, weight_mm

    def _forward_cross_self(self, proposals, track_queries, p_query_pos, t_query_pos, proposal_ref_pts, src, src_spatial_shapes,
                            level_start_index, num_proposals, src_padding_mask=None, attn_mask=None):
        if num_proposals != 0:
            # cross attention in proposals
            tmp_proposals = self.cross_attn(self.with_pos_embed(proposals, p_query_pos),
                                             proposal_ref_pts,
                                             src, src_spatial_shapes, level_start_index, src_padding_mask)
            tmp_proposals = proposals + self.dropout1(tmp_proposals)
            proposals = self.norm1(tmp_proposals)
        # ffn for proposals
        proposals = self.forward_ffn_proposal(proposals)
        # get proposal-track attention weights
        pt_weights_cos, pt_weights_mm = self._forward_weight_attn(proposals, p_query_pos, track_queries, t_query_pos, None)
        # get proposal-proposal attention weights
        pp_weights_cos, pp_weights_mm = self._forward_weight_attn(proposals, p_query_pos, proposals, p_query_pos)
        return proposals, pt_weights_cos, pt_weights_mm, pp_weights_cos, pp_weights_mm

    def forward(self, proposals, track_queries, p_query_pos, t_query_pos, proposal_ref_pts, src, src_spatial_shapes, num_proposals,
                level_start_index, src_padding_mask=None, mem_bank=None, mem_bank_pad_mask=None, attn_mask=None):
        if self.self_cross:
            raise NotImplementedError
        return self._forward_cross_self(proposals, track_queries, p_query_pos, t_query_pos, proposal_ref_pts, src, src_spatial_shapes,
                                        level_start_index, num_proposals, src_padding_mask, attn_mask)


def pos2posemb(pos, num_pos_feats=64, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    posemb = pos[..., None] / dim_t
    posemb = torch.stack((posemb[..., 0::2].sin(), posemb[..., 1::2].cos()), dim=-1).flatten(-3)
    return posemb


class SimDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                num_proposals, src_padding_mask=None, mem_bank=None, mem_bank_pad_mask=None, attn_mask=None):
        proposals = tgt[:, :num_proposals]
        track_queries = tgt[:, num_proposals:]
        proposal_ref_pts = reference_points[:, :num_proposals]
        track_query_ref_pts  = reference_points[:, num_proposals:]
        intermediate = []
        intermediate_proposal_ref_pts = []
        intermediate_pt_weights_cos = []
        intermediate_pp_weights_cos = []
        intermediate_pt_weights_mm = []
        intermediate_pp_weights_mm = []
        for lid, layer in enumerate(self.layers):
            if proposal_ref_pts.shape[-1] == 4:
                proposal_ref_pts_input = proposal_ref_pts[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert proposal_ref_pts.shape[-1] == 2
                proposal_ref_pts_input = proposal_ref_pts[:, :, None] * src_valid_ratios[:, None]
            p_query_pos = pos2posemb(proposal_ref_pts)
            t_query_pos = pos2posemb(track_query_ref_pts)
            proposals, pt_weights_cos, pt_weights_mm, pp_weights_cos, pp_weights_mm = \
                layer(proposals, track_queries, p_query_pos, t_query_pos, proposal_ref_pts_input, src, src_spatial_shapes,
                      num_proposals, src_level_start_index, src_padding_mask, mem_bank, mem_bank_pad_mask, attn_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](proposals)
                if proposal_ref_pts.shape[-1] == 4:
                    new_proposal_ref_pts = tmp + inverse_sigmoid(proposal_ref_pts)
                    new_proposal_ref_pts = new_proposal_ref_pts.sigmoid()
                else:
                    assert proposal_ref_pts.shape[-1] == 2
                    new_proposal_ref_pts = tmp
                    new_proposal_ref_pts[..., :2] = tmp[..., :2] + inverse_sigmoid(proposal_ref_pts)
                    new_proposal_ref_pts = new_proposal_ref_pts.sigmoid()
                proposal_ref_pts = new_proposal_ref_pts.detach()

            if self.return_intermediate:
                intermediate.append(proposals)
                intermediate_proposal_ref_pts.append(proposal_ref_pts)
                intermediate_pt_weights_cos.append(pt_weights_cos)
                intermediate_pp_weights_cos.append(pp_weights_cos)
                intermediate_pt_weights_mm.append(pt_weights_mm)
                intermediate_pp_weights_mm.append(pp_weights_mm)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_proposal_ref_pts), torch.stack(intermediate_pt_weights_cos), \
                   torch.stack(intermediate_pp_weights_cos), torch.stack(intermediate_pt_weights_mm), torch.stack(intermediate_pp_weights_mm)

        return proposals, proposal_ref_pts, pt_weights_cos, pp_weights_cos, pt_weights_mm, pp_weights_mm


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(True)
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=0,
        decoder_self_cross=not args.decoder_cross_self,
        sigmoid_attn=args.sigmoid_attn,
        extra_track_attn=args.extra_track_attn,
        memory_bank=False,
    )


