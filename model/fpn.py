#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 14:37:35 2020

@author: Ming Ming Zhang

Feature Pyramid Networks (FPN)
"""

import tensorflow as tf
import numpy as np


def backbone_fpn(resnet_stages, num_filters, P6=True):
    """
    Adds a 5 stages FPN to ResNet 50/101.

    Parameters
    ----------
    resnet_stages : list
        the output [C1,C2,C3,C4,C5] from backbone_resnet() with non-empty C5.
    num_filters : integer
        number of filters in all conv layers.
    P6 : boolean, optional
        whether or not create P6 of network, where is a stride 2 subsampling of
        P5. The default is True.

    Returns
    -------
    list
        feature maps [P2,P3,P4,P5,P6] at each level of the second pyramid.

    """
    _, C2, C3, C4, C5 = resnet_stages
    
    P5 = tf.keras.layers.Conv2D(num_filters, (1,1), name='fpn_c5p5')(C5)
    P4 = tf.keras.layers.Add(name='fpn_p4add')([
        tf.keras.layers.UpSampling2D((2,2), name='fpn_p5upsampled')(P5),
        tf.keras.layers.Conv2D(num_filters, (1,1), name='fpn_c4p4')(C4)])
    P3 = tf.keras.layers.Add(name='fpn_p3add')([
        tf.keras.layers.UpSampling2D((2,2), name='fpn_p4upsampled')(P4),
        tf.keras.layers.Conv2D(num_filters, (1,1), name='fpn_c3p3')(C3)])
    P2 = tf.keras.layers.Add(name='fpn_p2add')([
        tf.keras.layers.UpSampling2D((2,2), name='fpn_p3upsampled')(P3),
        tf.keras.layers.Conv2D(num_filters, (1,1), name='fpn_c2p2')(C2)])
    
    for p in [P2, P3, P4, P5]:
        assert p.shape[1] > 3 and p.shape[2] > 3, \
            'Image shape is too small to have FPN.'
    
    # 5 stages for each anchor scale in RPN from the original paper
    P2 = tf.keras.layers.Conv2D(
        num_filters, (3,3), padding='same', name='fpn_p2')(P2)
    P3 = tf.keras.layers.Conv2D(
        num_filters, (3,3), padding='same', name='fpn_p3')(P3)
    P4 = tf.keras.layers.Conv2D(
        num_filters, (3,3), padding='same', name='fpn_p4')(P4)
    P5 = tf.keras.layers.Conv2D(
        num_filters, (3,3), padding='same', name='fpn_p5')(P5)
    if P6:
        P6 = tf.keras.layers.MaxPooling2D(
            (1,1), strides=(2,2), name='fpn_p6')(P5)
    else:
        P6 = None
    
    return [P2, P3, P4, P5, P6]
        

def compute_backbone_shapes(config, image_shape):
    """
    Computes the spatial size of each stage of the backbone FPN.

    Parameters
    ----------
    config : class
        The custom configuration generated by a subclass of Config(object).
    image_shape : tuple
        (height, width, channels), image shape.

    Returns
    -------
    numpy array
        [num_stages, 2] where 2 is (height, width).

    """
    assert config.BACKBONE in ["resnet50", "resnet101"], \
        'Only support ResNet50\101'
    return np.array([
        [int(np.ceil(image_shape[0]/stride)), 
         int(np.ceil(image_shape[1]/stride))] 
        for stride in config.BACKBONE_STRIDES])    