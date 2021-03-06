"""
Created on Thu Nov 26 14:35:33 2020

@author: Ming Ming Zhang

Residual Networks (ResNet)
"""

# inspired by 
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

import tensorflow as tf


def identity_block(input_tensor, filters, stage, block, train_bn=True):
    """
    Builds an identity shortcut in a bottleneck building block of a ResNet.

    Parameters
    ----------
    input_tensor : tf tensor
        (batch_size, height, width, channels), input tensor.
    filters : list
        integers, number of filters in 3 conv layers at the main path, where
        last number is equal to channels.
    stage : integer
        in [2,5], used for generating layer names.
    block : string
        lowercase letter, used for generating layer names.
    train_bn : boolean, optional
        train or freeze batch norm layers. The default is True.

    Returns
    -------
    output_tensor : tf tensor
        (batch_size, height, width, channels), output tensor.

    """
    num_filters_1, num_filters_2, num_filters_3 = filters
    conv_prefix = 'res' + str(stage) + block + '_branch'
    bn_prefix = 'bn' + str(stage) + block + '_branch'
    
    x = tf.keras.layers.Conv2D(
        num_filters_1, (1,1), name=conv_prefix + '2a')(input_tensor)
    x = tf.keras.layers.BatchNormalization(
        name=bn_prefix + '2a')(x, training=train_bn)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(
        num_filters_2, (3,3), padding='same', name=conv_prefix + '2b')(x)
    x = tf.keras.layers.BatchNormalization(
        name=bn_prefix + '2b')(x, training=train_bn)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(
        num_filters_3, (1,1), name=conv_prefix + '2c')(x)
    x = tf.keras.layers.BatchNormalization(
        name=bn_prefix + '2c')(x, training=train_bn)
    
    x = tf.keras.layers.Add()([input_tensor, x])
    output_tensor = tf.keras.layers.Activation(
        'relu', name='res' + str(stage) + block + '_out')(x)
    return output_tensor


def conv_block(input_tensor, filters, stage, block, strides=(2,2), train_bn=True):
    """
    Builds a projection shortcut in a bottleneck block of a ResNet.

    Parameters
    ----------
    input_tensor : tf tensor
        (batch_size, height, width, channels), input tensor.
    filters : list
        integers, number of filters in 3 conv layers at the main path.
    stage : integer
        in [2,5], used for generating layer names.
    block : string
        lowercase letter, used for generating layer names.
    strides : tuple
        integer, conv strides.
    train_bn : boolean, optional
        train or freeze batch norm layers. The default is True.

    Returns
    -------
    output_tensor : tf tensor
        (batch_size, height//strides, width//strides, num_filters_3) where 
        num_filters_3 is the last number in filters, output tensor.

    """
    num_filters_1, num_filters_2, num_filters_3 = filters
    conv_prefix = 'res' + str(stage) + block + '_branch'
    bn_prefix = 'bn' + str(stage) + block + '_branch'
    
    x = tf.keras.layers.Conv2D(
        num_filters_1, (1,1), strides, name=conv_prefix + '2a')(input_tensor)
    x = tf.keras.layers.BatchNormalization(
        name=bn_prefix + '2a')(x, training=train_bn)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(
        num_filters_2, (3,3), padding='same', name=conv_prefix + '2b')(x)
    x = tf.keras.layers.BatchNormalization(
        name=bn_prefix + '2b')(x, training=train_bn)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(
        num_filters_3, (1,1), name=conv_prefix + '2c')(x)
    x = tf.keras.layers.BatchNormalization(
        name=bn_prefix + '2c')(x, training=train_bn)
    
    shortcut = tf.keras.layers.Conv2D(
        num_filters_3, (1,1), strides, name=conv_prefix + '1')(input_tensor)
    shortcut = tf.keras.layers.BatchNormalization(
        name=bn_prefix + '1')(shortcut, training=train_bn)
    
    x = tf.keras.layers.Add()([shortcut, x])
    output_tensor = tf.keras.layers.Activation(
        'relu', name='res' + str(stage) + block + '_out')(x)
    return output_tensor


def backbone_resnet(input_image, architecture, stage5=False, train_bn=True):
    """
    Builds a backbone ResNet.

    Parameters
    ----------
    input_image : tf tensor
        (batch_size, height, width, channels), input tensor.
    architecture : string
        ResNet architecture in {'resnet50', 'resnet101'}.
    stage5 : boolean, optional
        whether or not create stage5 of network. The default is False.
    train_bn : boolean, optional
        train or freeze batch norm layers. The default is True.

    Returns
    -------
    list
        feature maps at each stage.

    """
    assert architecture in ['resnet50', 'resnet101'], \
        'Only support ResNet50\101'
    
    # stage 1
    x = tf.keras.layers.ZeroPadding2D((3,3))(input_image)
    x = tf.keras.layers.Conv2D(64, (7,7), (2,2), name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_conv1')(x, training=train_bn)
    x = tf.keras.layers.Activation('relu')(x)
    C1 = x = tf.keras.layers.MaxPooling2D((3,3), (2,2), padding='same')(x)
    
    # stage 2
    x = conv_block(
        x, [64,64,256], stage=2, block='A', strides=(1,1), train_bn=train_bn)
    x = identity_block(x, [64,64,256], stage=2, block='B', train_bn=train_bn)
    C2 = x = identity_block(
        x, [64,64,256], stage=2, block='C', train_bn=train_bn)
    
    # stage 3
    x = conv_block(x, [128,128,512], stage=3, block='A', train_bn=train_bn)
    x = identity_block(x, [128,128,512], stage=3, block='B', train_bn=train_bn)
    x = identity_block(x, [128,128,512], stage=3, block='C', train_bn=train_bn)
    C3 = x = identity_block(
        x, [128,128,512], stage=3, block='D', train_bn=train_bn)
    
    # stage 4
    x = conv_block(x, [256,256,1024], stage=4, block='A', train_bn=train_bn)
    num_blocks = {'resnet50':5, 'resnet101':22}[architecture]
    for i in range(num_blocks):
        x = identity_block(
            x, [256,256,1024], stage=4, block=chr(66+i), train_bn=train_bn)
    C4 = x
    
    # stage 5
    if stage5:
        x = conv_block(x, [512,512,2048], stage=5, block='A', train_bn=train_bn)
        x = identity_block(
            x, [512,512,2048], stage=5, block='B', train_bn=train_bn)
        C5 = x = identity_block(
            x, [512,512,2048], stage=5, block='C', train_bn=train_bn)
    else:
        C5 = None
        
    return [C1, C2, C3, C4, C5]
