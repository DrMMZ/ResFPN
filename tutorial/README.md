**[Flower Photos Classification](https://github.com/DrMMZ/ResFPN/blob/main/tutorial/flower_photos.ipynb)**

This jupyter notebook uses the [TF flowers dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers) and demonstrates the use of [ResFPN](https://github.com/DrMMZ/ResFPN/tree/main/model). With the [pretrained resenet ImageNet weights](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5) and 5 epochs training, it can achieve 0.92 (+-2%) accuracy or 0.82 (+-3%) F1-score averaged over classes on 10% images.

----

**[COVID-19 Classification](https://github.com/DrMMZ/ResFPN/blob/main/tutorial/COVIDx.ipynb)**

The [COVIDx dataset](https://github.com/lindawangg/COVID-Net) is used in this jupyter notebook. The goal is to classify
COVID-19 in chest X-ray images. The notebook explores the dataset, and goes through preprocessing, training, visualization, evaluation and transfer learning by loading pretrained ResNet weights.
