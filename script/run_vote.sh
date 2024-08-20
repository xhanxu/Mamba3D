GPU_ID=0
PATH_CKPT= # path to ckpt

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
--config cfgs/finetune_scan_hardest.yaml \
--test \
--exp_name Mamba3D_hardest_vote \
--ckpts $PATH_CKPT
