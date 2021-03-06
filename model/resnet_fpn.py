# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:44:08 2021

@author: Ming Ming Zhang

Ensemble Model: ResNet + FPN
"""


import resnet, fpn

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, datetime


class ResFPN_Classifier():
    """
    An ensemble classifier from ResNet and FPN.
    
    """
    
    def __init__(self, image_shape, num_classes, num_filters=256, 
                 architecture='resnet50', augmentation=True):
        """
        Initialization.

        Parameters
        ----------
        image_shape : tuple
            [height, width, 3] where 3 is RBG channels, the image shape.
        num_classes : integer
            The number of classes in the given dataset.
        num_filters : integer, optional
            The number of filters used in FPN layers. The default is 256.
        architecture : string
            A ResNet architecture, either 'resnet50' or 'resnet101'. 
        augmentation : boolean, optional
            Whether to use keras augmentation, i.e., RandomFlip, RandomRotation,
            RandomZoom; see tf.keras.layers.experimental.preprocessing for 
            details. The default is True.

        Returns
        -------
        None.

        """
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.augmentation = augmentation
        self.model = self.build(image_shape, num_classes, num_filters, 
                                architecture, augmentation)
        
    
    def build(self, image_shape, num_classes, num_filters, architecture,
              augmentation):
        """
        Builds the classifier.

        Parameters
        ----------
        Same as the above.

        Returns
        -------
        model : keras model
            The ensemble classifier from ResNet and FPN.

        """
        inputs = tf.keras.Input(shape=image_shape, name='input_images')
        
        # augmentation & normalization
        if augmentation:
            x = tf.keras.layers.experimental.preprocessing.RandomFlip(
                mode='horizontal')(inputs)
            x = tf.keras.layers.experimental.preprocessing.RandomRotation(
                factor=0.1)(x)
            x = tf.keras.layers.experimental.preprocessing.RandomZoom(
                height_factor=0.1)(x)
            x = tf.keras.layers.experimental.preprocessing.Rescaling(
                scale=1.0/255)(x)
        else:
            x = tf.keras.layers.experimental.preprocessing.Rescaling(
                scale=1.0/255)(inputs)
            
        # resnet, either 'resnet50' or 'resnet101'
        C1, C2, C3, C4, C5 = resnet.backbone_resnet(
            x, architecture, stage5=True, train_bn=True)
        
        outputs = []
        
        # resnet head
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool_resnet')(C5)
        logits_resnet = tf.keras.layers.Dense(num_classes, name='dense_resnet')(x)
        outputs.append(logits_resnet)
        
        # fpn
        resnet_stages = [C1, C2, C3, C4, C5]
        P2, P3, P4, P5, P6 = fpn.backbone_fpn(
            resnet_stages, num_filters, P6=False)
        fmaps = [P2, P3, P4, P5]
        
        # fpn heads
        for i in range(len(fmaps)):
            p = fmaps[i]
            x = tf.keras.layers.GlobalAveragePooling2D(
                name='avg_pool_fpn_%s' % str(i+2))(p)
            logits_fpn = tf.keras.layers.Dense(
                num_classes, name='dense_fpn_%s' % str(i+2))(x)
            outputs.append(logits_fpn)
            
        
            
        model = tf.keras.Model(inputs, outputs, name=architecture + '_fpn')
        
        return model
    
    
    def compile(self, lr, momentum, l2):
        """
        Adds a SGD optimizer, L2-regularization and cross-entropy loss and gets
        ready for training.

        Parameters
        ----------
        lr : float
            A learning rate for the optimizer.
        momentum : float
            A momentum scalar for the optimizer. 
        l2 : float
            A scalar for the L2-regularization. 

        Returns
        -------
        None.

        """
        # optimizer
        optimizer = tf.keras.optimizers.SGD(
            lr=lr, momentum=momentum, clipnorm=5.0)
        
        # losses
        losses = []
        for i in range(len(self.model.outputs)):
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            losses.append(loss)
    
        # l2-regularization
        reg_losses = []
        for w in self.model.trainable_weights:
            reg_losses.append(
                tf.keras.regularizers.L2(l2)(w) / tf.cast(tf.size(w), tf.float32))
        self.model.add_loss(lambda: tf.math.add_n(reg_losses))
        
              
        self.model.compile(
            optimizer=optimizer,
            loss=losses,
            metrics=['accuracy'])
        
        
    def train(self, train_dataset, val_dataset, epochs, lr, momentum=0.9, 
              l2=0.01, save_weights=False):
        """
        Trains the model.

        Parameters
        ----------
        train_dataset : tf data
            A training dataset.
        val_dataset : tf data
            A validation dataset.
        lr : float
            A learning rate for the optimizer.
        momentum : float, optional
            A momentum scalar for the optimizer. The default is 0.9.
        l2 : float, optional
            A scalar for the L2-regularization. The default is 0.01.
        epochs : integer
            The number of training epochs.
        save_weights : boolean, optional
            Wether to save the trained weights in the working directory. The 
            default is False.

        Returns
        -------
        None.

        """
        self.compile(lr, momentum, l2)
        
        callbacks = []
        if save_weights:
            ROOT_DIR = os.getcwd()
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            checkpoint_dir = os.path.join(
                ROOT_DIR, 'checkpoints', current_time)
            self.checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint')
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                self.checkpoint_path, save_weights_only=True)
            callbacks.append(cp_callback)
            
        output = self.model.fit(
            train_dataset, validation_data=val_dataset, epochs=epochs, 
            callbacks=callbacks)
        self.history = output.history
        
        
    def plot(self):
        """
        Plots the learning curves.

        Returns
        -------
        None.

        """
        loss_names = []
        for x in self.history.keys():
            if 'loss' in x:
                loss_names.append(x)
        train_loss_names, val_loss_names = [], []
        for x in loss_names:
            if 'val_loss' in x:
                val_loss_names.append(x)
            elif x == 'loss':
                train_loss_names.append(x)
            
        acc_names = []
        for x in self.history.keys():
            if 'accuracy' in x:
                acc_names.append(x)
        train_acc_names, val_acc_names = [], []
        for x in acc_names:
            if 'val' in x:
                val_acc_names.append(x)
            else:
                train_acc_names.append(x)
    
        train_losses = []
        for name in train_loss_names:
            train_losses.append(self.history[name])
        val_losses = []
        for name in val_loss_names:
            val_losses.append(self.history[name])
        
        train_accs = []
        for name in train_acc_names:
            train_accs.append(self.history[name])
        val_accs = []
        for name in val_acc_names:
            val_accs.append(self.history[name])

        plt.subplots(2, 1, figsize=(15,12))

        plt.subplot(2, 1, 1)
        for i in range(len(train_loss_names)): 
            plt.plot(train_losses[i], label=train_loss_names[i])
        for i in range(len(val_loss_names)): 
            plt.plot(val_losses[i], label=val_loss_names[i])
        plt.xticks(range(len(train_losses[i])))
        plt.title('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        for i in range(len(train_acc_names)): 
            plt.plot(train_accs[i], label=train_acc_names[i])
        for i in range(len(val_acc_names)): 
            plt.plot(val_accs[i], label=val_acc_names[i])
        plt.xticks(range(len(train_accs[i])))
        plt.title('Accuracy')
        plt.legend()

        plt.show()

            
    def select_top(self, val_dataset, top):
        """
        Selects the top classifiers based on the accuracies.

        Parameters
        ----------
        val_dataset : tf data
            A validation dataset.
        top : integer
            The total number in [1, 6] of classifiers to select.

        Returns
        -------
        top_idxes : numpy array
            The indices of top classifiers based on the result in validation 
            dataset.
        ensemble_acc : float
            The accuracy of the model selected from top_idxes classifiers.

        """
        evals = np.array(self.model.evaluate(val_dataset))
        top_idxes = np.argsort(evals[7:])[::-1][:top]
        self.top_idxes = top_idxes
        
        # the ensemble model outputs, len(logits) = top
        logits = self.model.predict(val_dataset)
        logits = [logits[i] for i in top_idxes]
        
        # predicted class IDs, resulting in [num_val, top]
        class_ids = []
        for logit in logits:
            class_ids.append(np.argmax(logit, axis=1))
        class_ids = np.stack(class_ids, axis=1)
        
        # predicted class IDs, resulting in [num_val]
        num_val = class_ids.shape[0]
        ensemble_class_ids = np.zeros(shape=(num_val), dtype=np.int32)
        for i in range(num_val):
            unique_class_ids, counts = np.unique(
                class_ids[i], return_counts=True)
            idx = np.argmax(counts)
            class_id = unique_class_ids[idx]
            ensemble_class_ids[i] = class_id
            
        # accuracy
        ensemble_accs = []
        for x_batch, y_batch in val_dataset.take(1):
            batch_size = y_batch.shape[0]
        num_batches = int(np.ceil(num_val / batch_size))
        for i, (_, y_batch) in zip(
                range(num_batches), val_dataset.as_numpy_iterator()):
            y_gt_batch = ensemble_class_ids[i*batch_size:batch_size+i*batch_size]
            acc = np.sum(y_gt_batch == y_batch) / len(y_batch)
            ensemble_accs.append(acc)
        ensemble_acc = sum(ensemble_accs) / len(ensemble_accs)
        
        return top_idxes, ensemble_acc
    
    
    def predict(self, test_dataset):
        """
        Predicts the given dataset.

        Parameters
        ----------
        test_dataset : tf data
            The data needed to be predicted.

        Returns
        -------
        ensemble_class_ids : 
            The predicted class IDs using the ensemble model from select_top().

        """
        # the ensemble model outputs, len(logits) = top
        logits = self.model.predict(test_dataset)
        logits = [logits[i] for i in self.top_idxes]
        
        # predicted class IDs, resulting in [num_test, top]
        class_ids = []
        for logit in logits:
            class_ids.append(np.argmax(logit, axis=1))
        class_ids = np.stack(class_ids, axis=1)
        
        # predicted class IDs, resulting in [num_test]
        num_test = class_ids.shape[0]
        ensemble_class_ids = np.zeros(shape=(num_test), dtype=np.int32)
        for i in range(num_test):
            unique_class_ids, counts = np.unique(
                class_ids[i], return_counts=True)
            idx = np.argmax(counts)
            class_id = unique_class_ids[idx]
            ensemble_class_ids[i] = class_id
            
        return ensemble_class_ids

