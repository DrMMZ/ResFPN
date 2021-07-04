# ResFPN

This is an implementation of [*ResFPN*](https://github.com/DrMMZ/ResFPN/tree/main/model) on Python 3 and TensorFlow 2. The model classifies images by ensembling predictions from [Residual Network](https://arxiv.org/abs/1512.03385) (ResNet) and [Feature Pyramid Network](https://arxiv.org/abs/1612.03144) (FPN), and can be trained by minimizing [focal loss](https://arxiv.org/abs/1708.02002). 

The repository includes:
* source code of ResFPN built on ResNet50/101 and FPN;
* source code of focal loss (generalize to multi-class, with class balancing parameter); and
* jupyter notebook demonstration using ResFPN in training, evaluation and visualization on the [tf_flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers) and [COVIDx](https://github.com/lindawangg/COVID-Net) dataset. Below are example classifications on the tf_flowers dataset.

![tf_flowers](https://raw.githubusercontent.com/DrMMZ/drmmz.github.io/master/images/flower_movie.gif)


### Requirements
`python 3.7.9`, `tensorflow 2.3.1`, `matplotlib 3.3.4` and `numpy 1.19.2`

### Updates
* 07/04/2021: Add synchronized SGD over multiple GPUs training, and some callbacks such as CSVLogger, ModelCheckpoint and ReduceLROnPlateau. Finally, modify the functions resnet_fpn.select_top() and resnet_fpn.predict() to have ability to visualize the predictions. A new notebook on the tf_flower dataset are presented as a demonstration.
* 05/10/2021: Add [Focal Loss](https://arxiv.org/abs/1708.02002) implementation and some corresponding changes in ResFPN are made, see the [model](https://github.com/DrMMZ/ResFPN/tree/main/model) folder for details. Roughly speaking, focal loss can address class imbalance problem by removing easy examples during training. We present experimental results on the [COVIDx](https://github.com/lindawangg/COVID-Net) dataset, see the [tutorial](https://github.com/DrMMZ/ResFPN/tree/main/tutorial) folder.
