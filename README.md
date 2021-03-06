# ResFPN

This is an implementation of *ResFPN* on Python 3 and TensorFlow 2. The model classifies images by ensembling predictions from [Residual Network](https://arxiv.org/abs/1512.03385) (ResNet) and [Feature Pyramid Network](https://arxiv.org/abs/1612.03144) (FPN). 

The repository includes:
* source code of ResFPN built on ResNet50/101 and FPN, shown in the [model](https://github.com/DrMMZ/ResFPN/tree/main/model) folder; and
* jupyter notebook demonstration the use of ResFPN in training, evaluation and visualization, shown in the [tutorial](https://github.com/DrMMZ/ResFPN/tree/main/tutorial) folder.

### Requirements
`python 3.7.9`, `tensorflow 2.3.1`, `matplotlib 3.3.4` and `numpy 1.19.2`
