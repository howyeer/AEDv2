export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_DISTRIBUTED_TIMEOUT=10000
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
taskset -c 22-25 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29506 \
    --use_env aedv2/main.py \
    -c aedv2/config/GroundingDINO_SwinT_OGC_AED.py \
    -p weights/groundingdino_swint_ogc.pth \
    --meta_arch 'AED' \
    --dataset_file 'tao' \
    --epochs 2 \
    --print_freq 50 \
    --with_box_refine  \
    --lr_drop 1 \
    --lr 0.001 \
    --batch_size 1 \
    --sampler_lengths 8 \
    --decoder_cross_self \
    --clip_max_norm 10 \
    --save_period 1 \
    --mot_path /data/fzm_2022/Datasets \
    --bbox_loss_coef 0.5 \
    --giou_loss_coef 0.3 \
    --pt_weight_loss_coef 2 \
    --pp_weight_loss_coef 0.1 \
    --cross_clip_weight_loss_coef 1 \
    --train_score_thresh 0.3 \
    --train_base \
    --add_extra_dets \
    --random_drop 0 \
    --dec_n_points 4 \
    --match_high_score 0.3 \
    --val_match_high_thresh 0.5 \
    --val_match_low_thresh 0.5 \
    --train_match_thresh 0.5 \
    --val_max_det_num 100 \
    --train_max_det_num 200 \
    --train_iou_thresh 0.5 \
    --ema_weight 0.5 \
    --sample_interval 3 \
    --clip_gap 1 \
    --output_dir /home/hhy_2023/aaaacode/GroundingDINO/aedv2/output_dir
    


