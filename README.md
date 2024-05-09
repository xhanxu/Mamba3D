# Mamba3D
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mamba3d-enhancing-local-features-for-3d-point/supervised-only-3d-point-cloud-classification)](https://paperswithcode.com/sota/supervised-only-3d-point-cloud-classification?p=mamba3d-enhancing-local-features-for-3d-point)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mamba3d-enhancing-local-features-for-3d-point/3d-point-cloud-classification-on-modelnet40)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40?p=mamba3d-enhancing-local-features-for-3d-point)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mamba3d-enhancing-local-features-for-3d-point/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=mamba3d-enhancing-local-features-for-3d-point)

This repository contains the official implementation of the paper:

[**Mamba3D: Enhancing Local Features for 3D Point Cloud Analysis via State Space Model**](https://arxiv.org/abs/2404.14966)

- We present Mamba3D, a state space model tailored for point cloud learning.
<div style="text-align: center">
<img src="media/mamba3d_total_v2.png" />
</div>

- Mamba3D surpasses Transformer-based counterparts and concurrent works in multiple tasks, achieving multiple SoTA, with linear complexity.
<div style="text-align: center">
<img src="media/flops_v2.png"  />
</div>

## 📋 TODO
- [ ] Release the training and evaluation code
- [ ] Release the pretrained weights
- [ ] Release the toy code on Colab

## 😊 Acknowledgement
We would like to thank the authors of [Mamba](https://github.com/state-spaces/mamba), [Vision Mamba](https://github.com/hustvl/Vim), and [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) for their great works and repos.


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
