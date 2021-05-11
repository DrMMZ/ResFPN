# -*- coding: utf-8 -*-
"""
Created on Wed May  5 21:28:16 2021

@author: Ming Ming Zhang, mmzhangist@gmail.com

Focal Loss
"""

import tensorflow as tf


def focal_loss(
        y_true, 
        y_pred, 
        alpha=[1-0.25, 0.25], 
        gamma=2.0, 
        from_logits=False
        ):
    """
    Computes the multi-class focal loss, with class balancing parameter.

    Parameters
    ----------
    y_true : tensor
        Targets of shape [num_targets,].
    y_pred : tensor
        Predictions of shape [num_targets, num_classes].
    alpha : list
        Weighting factors for classes, addressing class imbalance. For example 
        in binary situation, [1-0.25, 0.25] means 0.75 for negative class and 
        0.25 for positive class.
    gamma : float, optional
        A focusing parameter >= 0 for removing easy examples. The default is 2.
    from_logits : boolean, optional
        Whether y_pred is a logits tensor. The default is False, i.e., 
        probability.

    Returns
    -------
    loss : tensor
        Focal loss, a scalar.

    """
    # multi-class cross-entropy loss
    ce = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=from_logits)
    
    # softmax estimated probability
    if from_logits:
        pred_prob = tf.math.softmax(y_pred)
    else:
        pred_prob = y_pred
        
    # corresponding t-th probability to y_true
    idxes = tf.stack([
        tf.range(tf.shape(y_pred)[0]), 
        tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)
        ], axis=1)
    p_t = tf.gather_nd(pred_prob, idxes)
    
    # modulating factor
    gamma = tf.convert_to_tensor(gamma, dtype=tf.keras.backend.floatx())
    modulating_factor = tf.math.pow((1.0 - p_t), gamma)
    
    # alpha factor
    alpha = tf.convert_to_tensor(alpha, dtype=tf.keras.backend.floatx())
    alpha_factor = tf.gather(alpha, tf.squeeze(tf.cast(y_true, tf.int32)))
    
    # focal loss
    loss = alpha_factor * modulating_factor * ce
    
    # reduce to scalar
    loss = tf.math.reduce_mean(loss, axis=-1)
    return loss


class FocalLoss(tf.keras.losses.Loss):
    """
    Defines a class focal loss as a subclass of TF loss.
    
    """
    def __init__(self, alpha, gamma=2.0, from_logits=False):
        super().__init__(name='focal_loss')
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        
    def call(self, y_true, y_pred):
        loss = focal_loss(
            y_true, 
            y_pred, 
            alpha=self.alpha, 
            gamma=self.gamma, 
            from_logits=self.from_logits
            )
        return loss
