# Mamba3D 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mamba3d-enhancing-local-features-for-3d-point/supervised-only-3d-point-cloud-classification)](https://paperswithcode.com/sota/supervised-only-3d-point-cloud-classification?p=mamba3d-enhancing-local-features-for-3d-point)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mamba3d-enhancing-local-features-for-3d-point/3d-point-cloud-classification-on-modelnet40)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40?p=mamba3d-enhancing-local-features-for-3d-point)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mamba3d-enhancing-local-features-for-3d-point/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=mamba3d-enhancing-local-features-for-3d-point)

This repository contains the official implementation of the paper:
  
[**[ACM MM 24] Mamba3D: Enhancing Local Features for 3D Point Cloud Analysis via State Space Model**](https://arxiv.org/abs/2404.14966)

 
## 📰 News
- [2024/8] After optimizing the code and the model, Mamba3D can now achieve an overall accuracy of 92.05% on the ScanObjectNN(PB_T50_RS) dataset! We have also updated the results in the paper [here](https://arxiv.org/abs/2404.14966).
- [2024/8] We release the pretrained weights [here](https://huggingface.co/hanx/Mamba3D)!
- [2024/8] We release the training and evaluation code! Pretrained weights are coming soon!
- [2024/7] Our [MiniGPT-3D](https://github.com/tangyuan96/minigpt-3d) is also accepted by ACM MM24! We outperform existing large point-language models, using just about 1 day on 1 RTX 3090! Check it out!
- [2024/7] Ours Mamba3D is accepted by ACM MM24!

- [2024/4] We present Mamba3D, a state space model tailored for point cloud learning.
<div style="text-align: center">
<img src="media/mamba3d_total_v2.png" />
</div>


## 📋 TODO
- [x] Release the training and evaluation code
- [x] Release the pretrained weights
- [ ] Release the toy code on Colab


## 🎒 1. Requirements
Tested on:
PyTorch == 1.13.1;
python == 3.8;
CUDA == 11.7

```
pip install -r requirements.txt
```

```
# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# Mamba install
pip install causal-conv1d==1.1.1
pip install mamba-ssm==1.1.1
```

More detailed settings can be found in [mamba3d.yaml](./mamba3d.yaml).

## 🧾 2. Datasets & Pretrained Weights

We use ShapeNet, ScanObjectNN, ModelNet40 and ShapeNetPart in this work. See [DATASET.md](./DATASET.md) for details.

You can find the pre-trained weights [here](https://huggingface.co/hanx/Mamba3D).
Or, specifically as follows.
| Dataset             | Pretrain | Acc   | Weight |
|---------------------|----------|-------|--------|
|ShapeNet             | Point-MAE|    |[ckpt](https://huggingface.co/hanx/Mamba3D/blob/main/pretrain_pointmae/ckpt-last.pth)   |  
| ModelNet40          | no       | 93.4  | [ckpt](https://huggingface.co/hanx/Mamba3D/blob/main/modelnet40_scratch_93.4/ckpt-best.pth)    |
| ModelNet40          | Point-MAE | 94.7  | [ckpt](https://huggingface.co/hanx/Mamba3D/blob/main/modelnet40_pointmae_94.7/ckpt-best.pth)    |
| ScanObjectNN-hardest| no       | 91.81 | [ckpt](https://huggingface.co/hanx/Mamba3D/blob/main/scanobjectnn_hardest_scratch_91.81/ckpt-best-91.8.pth)    |
| ScanObjectNN-hardest| Point-MAE | 92.05 | [ckpt](https://huggingface.co/hanx/Mamba3D/blob/main/scanobjectnn_hardest_pointmae_92.05/ckpt-best.pth)    |

## 🥧 3. Training from scratch

To train Mamba3D on ScanObjectNN/Modelnet40 from scratch, run:
```
# Note: change config files for different dataset
bash script/run_scratch.sh
```

To vote on ScanObjectNN/Modelnet40, run:
```
# Note: change config files for different dataset
bash script/run_vote.sh
```
Few-shot learning, run:
```
bash script/run_fewshot.sh
```

<!-- ## 🐟 4. Pretraining & Finetuning -->
## 🐟 4. Finetuning
<!-- To pre-train Mamba3D on ShapeNet, run:

```
# Note: change config files for different dataset
bash script/run_pretrain.sh
# or
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/pretrain.yaml --exp_name <output_file_name>
``` -->
To fine-tune Mamba3D on ScanObjectNN/Modelnet40, run:
```
# Note: change config files for different dataset
bash script/run_finetune.sh
```

## 😊 Acknowledgement
We would like to thank the authors of [Mamba](https://github.com/state-spaces/mamba), [Vision Mamba](https://github.com/hustvl/Vim), and [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) for their great works and repos.

## 😀 Contact
If you have any questions or are looking for cooperation in related fields, please contact [Xu Han](https://xhanxu.github.io/) via xhanxu@hust.edu.cn. 

## 📚 Citation
If you find our work helpful, please consider citing:
```bibtex
@article{han2024mamba3d,
  title={Mamba3D: Enhancing Local Features for 3D Point Cloud Analysis via State Space Model},
  author={Han, Xu and Tang, Yuan and Wang, Zhaoxuan and Li, Xianzhi},
  journal={arXiv preprint arXiv:2404.14966},
  year={2024}
}
```
