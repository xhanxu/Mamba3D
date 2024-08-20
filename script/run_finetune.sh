GPU_ID=1
PATH_CKPT= # path to ckpt

# ScanObjectNN
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
--config cfgs/finetune_scan_objonly.yaml \
--finetune_model \
--exp_name Mamba3D_objonly_finetune \
--ckpts $PATH_CKPT

# ModelNet40 1K
# CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
# --config cfgs/finetune_modelnet.yaml \
# --finetune_model \
# --exp_name Mamba3D_modelnet_finetune \
# --ckpts $PATH_CKPT