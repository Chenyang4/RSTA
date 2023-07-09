# ReSmooth with TeachAugment strategy
This repository contains the experimental code for TeachAugment result in TABLE III.

To be noticed, for fair comparison and minimal modification, we don't interfere the augmentation optimization process of TeachAugment by externally training a RS model. (It feels reasonable to optimize the augmentation directly from the RS model or only keep the final augmentation in the buffer for ReSmooth. But it's not the case of this repository.)

More details about ReSmooth and TeachAugment can be found in [References](#References).

## Requirements

```tex
python=3.9
pytorch>=1.8.1
torchvision>=0.9.1
skimage
sklearn
tqdm
matplotlib
tensorboard
git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
```

## Experiments

To reproduce the result of TeachAugment in TABLE III, one can run the following commands. 

Data will be downloaded automatically. The pretrained model for RS is provided [here](https://drive.google.com/file/d/1WVUsxUsRPpa6UEnr9MwqR8TwigOtgJMh/view?usp=sharing) and should be placed in the project directory.

```bash
# TeachAugment result
python main.py --yaml ./config/CIFAR100/resnet18.yaml --log_dir log/ta
python main.py --yaml ./config/CIFAR100/resnet18.yaml --ls --smooth-org 0.3 --log_dir log/ls
python main.py --yaml ./config/CIFAR100/resnet18.yaml --smooth-aug 0.5 --gmm --pretrained model.pth --log_dir log/rs
```

## Citation

If you find the repository useful for your research, please cite ReSmooth and TeachAugment as follows:

```
@article{wang2022resmooth,
  title={ReSmooth: Detecting and Utilizing OOD Samples When Training With Data Augmentation},
  author={Wang, Chenyang and Jiang, Junjun and Zhou, Xiong and Liu, Xianming},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022},
  publisher={IEEE}
}
@inproceedings{suzuki2022teachaugment,
  title={Teachaugment: Data augmentation optimization using teacher knowledge},
  author={Suzuki, Teppei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10904--10914},
  year={2022}
}
```

## References 

- ReSmooth: [Paper](https://ieeexplore.ieee.org/abstract/document/9961105) [Code](https://github.com/Chenyang4/ReSmooth)
- TeachAugment: [Paper](https://arxiv.org/abs/2202.12513) [Code](https://github.com/DensoITLab/TeachAugment)
