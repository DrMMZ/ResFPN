# ResFPN

This is an implementation of [*ResFPN*](https://github.com/DrMMZ/ResFPN/tree/main/model) on Python 3 and TensorFlow 2. The model classifies images by ensembling predictions from [Residual Network](https://arxiv.org/abs/1512.03385) (ResNet) and [Feature Pyramid Network](https://arxiv.org/abs/1612.03144) (FPN). 

The repository includes:
* source code of ResFPN built on ResNet50/101 and FPN;
* source code of [Focal Loss](https://github.com/DrMMZ/ResFPN/blob/main/model/focal_loss.py) (generalize to multi-class, with class balancing parameter); and
* jupyter notebook demonstration the use of ResFPN in training, evaluation and visualization.

### Requirements
`python 3.7.9`, `tensorflow 2.3.1`, `matplotlib 3.3.4` and `numpy 1.19.2`

### Updates
* 05/10/2021: Add [Focal Loss](https://arxiv.org/abs/1708.02002) implementation and some corresponding changes in ResFPN are made, see the [model](https://github.com/DrMMZ/ResFPN/tree/main/model) folder for details. Roughly speaking, focal loss can address class imbalance problem by removing easy examples during training. We present experimental results on the [COVIDx](https://github.com/lindawangg/COVID-Net) dataset, see the [tutorial](https://github.com/DrMMZ/ResFPN/tree/main/tutorial) folder.
