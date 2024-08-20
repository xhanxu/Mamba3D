GPU_ID=0
PATH_CKPT= # path to ckpt

# --way <5 or 10> --shot <10 or 20>

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
--config cfgs/fewshot.yaml \
--fewshot_model \
--exp_name Mamba3D_fewshot \
--ckpts $PATH_CKPT \
--way 5 \
--shot 10