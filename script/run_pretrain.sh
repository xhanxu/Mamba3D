GPU_ID=0

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
    --config cfgs/pretrain.yaml \
    --exp_name Mamba3D_pointmae