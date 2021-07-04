#### Updates
* 07/04/2021: Add a new notebook using the updated ResFPN on Flower dataset.
* 05/10/2021: Due to a new version of ResFPN, we made corresponding changes in the demonstrations using Flower and COVIDx dataset. Old jupyter notebooks are in the [bin](https://github.com/DrMMZ/ResFPN/tree/main/tutorial/bin) folder.

----

**[Flower Photos Classification](https://github.com/DrMMZ/ResFPN/blob/main/tutorial/flower_photos.ipynb)**

This jupyter notebook uses the [TF flowers dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers) and demonstrates the use of [ResFPN](https://github.com/DrMMZ/ResFPN/tree/main/model). With the [pretrained resenet ImageNet weights](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5), 5 epochs synchronized SGD over 2 GPUs training on augmented 256x256 images a batch size of 64 (32 images per GPU), it can achieve 0.92 (+-2%) accuracy or 0.83 (+-3%) F1-score averaged over classes on 10% images.

----

**[COVID-19 Classification](https://github.com/DrMMZ/ResFPN/blob/main/tutorial/COVIDx.ipynb)**

The [COVIDx dataset](https://github.com/lindawangg/COVID-Net) is used in this jupyter notebook. The goal is to classify
COVID-19 in chest X-ray images. The notebook presents experimental results on the cross-entropy loss and [focal loss](https://github.com/DrMMZ/ResFPN/tree/main/model). For preprocessing the dataset and tuning parameters in focal loss, see [covidx](https://github.com/DrMMZ/ResFPN/blob/main/tutorial/covidx.py) file.

With the focal loss and [pretrained resenet ImageNet weights](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5), 5 epochs SGD training on 256x256 images a batch size of 32, it can achieve 0.77 (+-1%) F1-score averaged over classes on the test set (one with 100 COVID-19 images). On the contrast, cross-entropy loss has 0.72 (+-2%) F1-score averaged over classes.
