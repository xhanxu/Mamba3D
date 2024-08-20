GPU_ID=0

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
--config cfgs/finetune_scan_hardest.yaml \
--scratch_model \
--exp_name Mamba3D_hardest_scratch