batch_size = 1
modelname = "groundingaed"
backbone = "swin_T_224_1k"
position_embedding = "sine"
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
two_stage_type = "standard"
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
transformer_activation = "relu"
dec_pred_bbox_embed_share = True
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
dn_label_coef = 1.0
dn_bbox_coef = 1.0
embed_init_tgt = True
dn_labelbook_size = 2000
max_text_len = 256
text_encoder_type = "/home/hhy_2023/aaaacode/GroundingDINO/bert-base-uncased"
use_text_enhancer = True
use_fusion_layer = True
use_checkpoint = True
use_transformer_ckpt = True
use_text_cross_attention = True
text_dropout = 0.0
fusion_dropout = 0.0
fusion_droppath = 0.1
sub_sentence_present = True
meta_arch = "AED"
dataset_file = "tao"
epochs = 5
print_freq = 500
with_box_refine = True
lr_drop = 2
lr = 1e-4
lr_linear_proj_mult = 1
lr_backbone = 1e-4
lr_backbone_names = ["backbone.0"]
lr_linear_proj_names =['reference_points', 'sampling_offsets',]
pretrained = "/home/hhy_2023/aaaacode/AED/pretrained/tao_ckpt_train_base.pth"
batch_size = 1
sampler_lengths = [5]
decoder_cross_self = True
clip_max_norm = 10
save_period = 9
mot_path = "/data/fzm_2022/Datasets"
bbox_loss_coef = 0.5
giou_loss_coef = 0.3
pt_weight_loss_coef = 2
pp_weight_loss_coef = 0.1
cross_clip_weight_loss_coef = 1
val_nms_thresh = 0.4
val_score_thresh = 0.3
train_score_thresh = 0.3
train_base = True
add_extra_dets = True
random_drop = 0
dec_n_points = 4
match_high_score = 0.3
val_match_high_thresh = 0.5
val_match_low_thresh = 0.5
train_match_thresh = 0.5
val_max_det_num = 100
train_max_det_num = 200
train_iou_thresh = 0.5
ema_weight = 0.5
sample_interval = 3
clip_gap = 1
occupy_mem = False
device = "cuda"
seed = 42
aux_loss = True
sample_mode ='random_interval' # choices=('fixed_interval', 'random_interval'),
num_workers = 4