export CUDA_VISIBLE_DEVICES=3
python3 demo/inference_on_a_image.py \
    -c groundingdino/config/GroundingDINO_SwinT_OGC.py \
    -p weights/groundingdino_swint_ogc.pth \
    -i /data/fzm_2022/Datasets/TAO/frames/train/HACS/Shoveling_snow_v_qGbvHecNfEo_scene_0_0-902/frame0151.jpg \
    -o /home/hhy_2023/aaaacode/GroundingDINO/output \
    -t "dog . person . shovel"