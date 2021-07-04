"""
Created on Thu Mar  4 13:44:08 2021

@author: Ming Ming Zhang, mmzhangist@gmail.com

Ensemble Model: ResNet + FPN
"""


import resnet, fpn, focal_loss

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, datetime


class ResFPN_Classifier():
    """
    An ensemble classifier from ResNet and FPN.
    
    """
    
    def __init__(
            self, 
            image_shape, 
            num_classes, 
            architecture='resnet50', 
            augmentation=False, 
            train_bn=False,
            params={'lr':1e-3, 'l2':1e-4, 'epochs':2}, 
            loss_type='ce', 
            save_weights=True,
            reduce_lr=False,
            checkpoint_path=None, 
            resnet_weights_path=None,
            multi_gpu_training=False,
            ):
        """
        A constructor.

        Parameters
        ----------
        image_shape : tuple, [height, width, 3] where 3 is RBG channels
            The shape of images.
        num_classes : integer
            The number of classes in the given dataset.
        architecture : string
            A ResNet architecture, either 'resnet50' or 'resnet101'. 
        augmentation : boolean, optional
            Whether to use augmentation, i.e., RandomFlip, RandomContrast, 
            RandomRotation, RandomZoom. The default is True. see 
            tf.keras.layers.experimental.preprocessing for details. 
        train_bn : boolean, optional
            Whether one should normalize the layer input by the mean and 
            variance over the current batch. The default is False, i.e., use 
            the moving average of mean and variance to normalize the layer 
            input.
        params : dictionary, optional
            Training parameters including the learning rate lr, momentum, 
            L2-regularization l2, epochs, alpha and gamma parameters when using 
            focal loss. The default is {'lr':1e-3, 'l2':1e-4, 'epochs':2}.
        loss_type : string, optional
            Whether to use cross-entropy 'ce' or focal loss 'focal'. The default 
            is 'ce'.
        save_weights : boolean, optional
            Whether to save the trained weights in the working directory. The 
            default is False.
        reduce_lr : boolean, optional
            Whether to reduce learning rate by a factor 0.1 after 10 epochs no 
            decreased val_loss.
        checkpoint_path : string, optional
            The path to the saved TF keras model checkpoint. The default is 
            None.
        resnet_weights_path : string, optional
            The path to the pretrained ResNet weights in h5 format. Note that
            it has to match with the architecture of building model. The default 
            is None.
        multi_gpu_training : boolean, optional
            Whether to use multi-GPU training if one has multiple GPUs. The 
            default is False.

        Returns
        -------
        None.

        """
        self.num_classes = num_classes
        self.train_bn = train_bn
        self.params = params 
        self.loss_type = loss_type, 
        self.save_weights = save_weights,
        self.reduce_lr = reduce_lr,
        
        if multi_gpu_training and len(tf.config.list_physical_devices('GPU'))>1:
            strategy = tf.distribute.MirroredStrategy(
                cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        else:
            strategy = tf.distribute.get_strategy() 
            
        with strategy.scope():
            self.model = self.build(
                image_shape, 
                num_classes, 
                architecture, 
                augmentation,
                train_bn)
            
            if checkpoint_path is not None:
                print('\nLoading checkpoint:\n%s\n' \
                      % checkpoint_path)
                self.model.load_weights(checkpoint_path, by_name=False)
                
            if resnet_weights_path is not None:
                print('\nLoading resnet:\n%s\n' \
                          % resnet_weights_path)
                self.model.load_weights(
                    resnet_weights_path, by_name=True)
                        
            self.compile_model(self.model, params, loss_type)
        
    
    def build(
            self, 
            image_shape, 
            num_classes, 
            architecture,
            augmentation,
            train_bn
            ):
        """
        Builds the ResFPN classifier.

        Parameters
        ----------
        Same as the above.

        Returns
        -------
        model : tf keras model
            An ensemble classifier from ResNet and FPN.

        """
        inputs = tf.keras.Input(shape=image_shape, name='input_images')
        
        # augmentation & normalization /255
        if augmentation:
            x = tf.keras.layers.experimental.preprocessing.RandomFlip(
                mode='horizontal')(inputs)
            x = tf.keras.layers.experimental.preprocessing.RandomContrast(
                factor=0.5)(x)
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
            x, architecture, stage5=True, train_bn=train_bn)
        
        outputs = []
        
        # resnet head
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool_resnet')(C5)
        logits_resnet = tf.keras.layers.Dense(
            num_classes, name='dense_resnet')(x)
        outputs.append(logits_resnet)
        
        # fpn
        resnet_stages = [C1, C2, C3, C4, C5]
        P2, P3, P4, P5, P6 = fpn.backbone_fpn(
            resnet_stages, 
            num_filters=256, 
            P6=False)
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
    
    
    def compile_model(self, model, params={}, loss_type='ce'):
        """
        Adds a SGD optimizer, L2-regularization and cross-entropy/focal loss.

        Parameters
        ----------
        model : tf keras model
            A ResFPN model.
        params, loss_type : same as above.

        Returns
        -------
        None.

        """
        # optimizer
        lr = params.get('lr', 1e-4)
        momentum = params.get('momentum', 0.9)
        l2 = params.get('l2', 1e-4)
        optimizer = tf.keras.optimizers.SGD(lr, momentum)
        
        # losses
        assert loss_type in ['ce', 'focal']
        losses = []
        for i in range(len(model.outputs)):
            if loss_type == 'ce':
                loss = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True)
            else:
                alpha = params.get(
                    'alpha', [1/self.num_classes] * self.num_classes)
                gamma = params.get('gamma', 1.0)
                loss = focal_loss.FocalLoss(
                    alpha=alpha,
                    gamma=gamma,
                    from_logits=True)
            losses.append(loss)
    
        # l2-regularization
        reg_losses = []
        for w in model.trainable_weights:
            # batchnorm weights don't contribute to the loss
            if 'gamma' not in w.name and 'beta' not in w.name:
                reg_losses.append(
                    tf.math.divide(
                        tf.keras.regularizers.L2(l2)(w),
                        tf.cast(tf.size(w), w.dtype)))
        model.add_loss(lambda: tf.math.add_n(reg_losses))
            
        model.compile(
            optimizer=optimizer,
            loss=losses,
            metrics=['accuracy'])
        
        
    def train(
            self, 
            train_dataset, 
            val_dataset, 
            plot_training=True
            ):
        """
        Trains the model.

        Parameters
        ----------
        train_dataset : tf dataset
            A training dataset.
        val_dataset : tf dataset
            A validation dataset.

        Returns
        -------
        None.

        """
        # assign a learning rate after loading a checkpoint; otherwise it will
        # continue on the last learning rate in the checkpoint
        self.model.optimizer.lr.assign(self.params['lr'])
        print('\nlearning rate:', self.model.optimizer.lr.numpy(), '\n')
        
        # callbacks, including CSVLogger, ModelCheckpoint and ReduceLROnPlateau
        callbacks = []
        ROOT_DIR = os.getcwd()
        log_dir = os.path.join(ROOT_DIR, 'checkpoints')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = os.path.join(log_dir, current_time)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        if self.save_weights:
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint')
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, 
                save_weights_only=True)
            callbacks.append(cp_callback)
        if self.reduce_lr:
            reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.1, 
                patience=10)
            callbacks.append(reduce_lr_callback)
        log_filename = os.path.join(checkpoint_dir, '%s.csv' % current_time)
        log_callback = tf.keras.callbacks.CSVLogger(
            log_filename, 
            append=False)
        callbacks.append(log_callback)
        
        epochs = self.params.get('epochs', 1)
        output = self.model.fit(
            train_dataset, 
            validation_data=val_dataset, 
            epochs=epochs, 
            callbacks=callbacks)
        self.history = output.history
        
        if plot_training:
            loss_names = []
            for x in self.history.keys():
                if 'loss' in x:
                    loss_names.append(x)
                else:
                    print(x, self.history[x])
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
            # plt.xticks(range(len(train_accs[i])))
            plt.title('Accuracy')
            plt.legend()
            plt.savefig(os.path.join(checkpoint_dir, '%s.png' % current_time))
            plt.show()

            
    def select_top(self, val_dataset, class_names, top=3, verbose=True):
        """
        Selects the top classifiers based on the losses.

        Parameters
        ----------
        val_dataset : tf dataset
            A validation dataset.
        class_names : list
            The class names in the given dataset.
        top : integer, optional
            The number from 1 to 5. The default is 3, i.e., the best (minimum) 
            3 losses over all 5 losses.
        verbose : boolean, optional
            Whether to print out metrics, which include accuracy, precision, 
            recall and F1-score. The default is True. 

        Returns
        -------
        top_idxes : numpy array, [top, ]
            The indices of top classifiers based on the result in validation 
            dataset.
        metrics : tuple
            The ensembled classifier performance over batches, including 
            accuracy, precision, recall and F1-score, where last three metrics
            are dictionaries with different class IDs as keys.

        """
        # fix the order of elements in val_dataset
        val_dataset = val_dataset.cache()
        
        evals = np.array(self.model.evaluate(val_dataset))
        top_idxes = np.argsort(evals[1:6])[:top]
        
        # display top classifiers names
        classifier_names = ['resnet']
        for i in np.arange(5):
            classifier_names += ['res_fpn_%d' % (i+2)]
        print(
            '\nTop classifiers:', 
            [classifier_names[i] for i in top_idxes], 
            '\n')
        
        # the ensemble model outputs, len(logits) = top
        logits = self.model.predict(val_dataset)
        logits = [logits[i] for i in top_idxes]
        
        # predicted class IDs, resulting in [num_val, top]
        class_ids = []
        for logit in logits:
            class_ids.append(np.argmax(logit, axis=1))
        class_ids = np.stack(class_ids, axis=1)
        
        # predicted class IDs, resulting in [num_val, ]
        num_val = class_ids.shape[0]
        ensemble_class_ids = np.zeros(shape=(num_val), dtype=np.int32)
        for i in range(num_val):
            unique_class_ids, counts = np.unique(
                class_ids[i], 
                return_counts=True)
            idx = np.argmax(counts)
            class_id = unique_class_ids[idx]
            ensemble_class_ids[i] = class_id
        
        def compute_metrics(positive_class_id, y_true, y_pred):
            """
            Given a positive class, computes precision, recall and F1-score.

            Parameters
            ----------
            positive_class_id : integer
                A positive class ID.
            y_true : numpy array
                Ground-truth class IDs.
            y_pred : numpy array
                Predicted class IDs, same shape as y_true.

            Returns
            -------
            metrics : tuple
                Includes precision, recall and F1-score.

            """
            tp, fp, fn, tn = 0, 0, 0, 0
            for idx in range(len(y_pred)):
                if y_pred[idx] == positive_class_id and \
                    y_pred[idx] == y_true[idx]:
                        tp += 1
                elif y_pred[idx] == positive_class_id and \
                    y_pred[idx] != y_true[idx]:
                        fp += 1
                elif y_pred[idx] != positive_class_id and \
                    y_pred[idx] != y_true[idx]:
                        fn += 1
                elif y_pred[idx] != positive_class_id and \
                    y_pred[idx] == y_true[idx]:
                        tn += 1
            precision = tp / (tp + fp + 1e-5)
            recall = tp / (tp + fn + 1e-5)
            f1_score = 2 * precision * recall / (precision + recall + 1e-5)
            metrics = (precision, recall, f1_score)
            return metrics
        
        # compute metrics
        accs, precs, recalls, f1s = [], {}, {}, {}
        for _, y_batch in val_dataset.take(1):
            batch_size = y_batch.shape[0]
        num_batches = int(np.ceil(num_val / batch_size))
        for i, (_, y_true_batch) in zip(
                range(num_batches), val_dataset.as_numpy_iterator()):
            y_pred_batch = ensemble_class_ids[
                i*batch_size : batch_size + i*batch_size]
            acc_batch = np.sum(y_true_batch == y_pred_batch) / len(y_true_batch)
            accs.append(acc_batch)
            # for each class, compute precision, recall and F1-score
            for class_id in range(len(class_names)):
                prec_batch, recall_batch, f1_batch = compute_metrics(
                    class_id, y_true_batch, y_pred_batch)
                if class_names[class_id] not in precs.keys():
                    precs[class_names[class_id]] = []
                precs[class_names[class_id]].append(prec_batch)
                if class_names[class_id] not in recalls.keys():
                    recalls[class_names[class_id]] = []
                recalls[class_names[class_id]].append(recall_batch)
                if class_names[class_id] not in f1s.keys():
                    f1s[class_names[class_id]] = []
                f1s[class_names[class_id]].append(f1_batch)
        metrics = (accs, precs, recalls, f1s)
        
        # display metrics
        if verbose:
            print('\nAccuracy: %.2f\n' % (np.mean(np.array(accs))))
            precs_class, recalls_class, f1s_class = [], [], []
            for name in class_names:
                precs_class.append(np.mean(np.array(precs[name])))
                recalls_class.append(np.mean(np.array(recalls[name]))) 
                f1s_class.append(np.mean(np.array(f1s[name])))
                print('%s:' % (name))
                print('  precision %.2f, recall %.2f, F1-score %.2f' % (
                    precs_class[-1], recalls_class[-1], f1s_class[-1])) 
            print('\nAverage:')
            print('  precision %.2f, recall %.2f, F1-score %.2f' % \
                  (
                      np.mean(precs_class), 
                      np.mean(recalls_class), 
                      np.mean(f1s_class)
                      )
                  )
        
        return top_idxes, metrics
    
    
    def predict(
            self, 
            image, 
            top_idxes=[0],
            class_names=None
            ):
        """
        Predicts the given dataset.

        Parameters
        ----------
        image : tf tensor, shape = image_shape
            The image needed to be predicted.
        top_idxes : list, optional
            An output from select_top(). The default is resnet's prediction.
        class_names : list, optional
            The class names in the given dataset. The default is None. If it
            is not None, then the image will be displayed with the predicted
            probabilities over ensembled class ids.

        Returns
        -------
        ensemble_class_ids : numpy array
            The predicted class id using the ensemble model from select_top().

        """
        image1 = tf.expand_dims(image, axis=0)
        
        # the ensemble model outputs, len(logits) = top
        logits = self.model(image1)
        logits = [logits[i] for i in top_idxes]
        
        # predicted class IDs, [top, ]
        class_ids = []
        for logit in logits:
            class_ids.append(np.argmax(logit, axis=1))
        class_ids = np.array(class_ids)
        unique_class_ids, counts = np.unique(class_ids, return_counts=True)
        idx = np.argmax(counts)
        ensemble_class_id = unique_class_ids[idx]
        
        def vis(image, class_names, probs):
            """
            Displays the image with the predicted probabilities.

            Parameters
            ----------
            image, class_names : same as above.
            probs : numpy array, [top, num_classes]
                The predicted probabilities over ensembled class ids.

            Returns
            -------
            None.

            """
            plt.imshow(tf.cast(image, tf.int32))
            for i in range(len(class_names)):
                if i == np.argmax(probs):
                    plt.text(
                        image.shape[1] + 10, 
                        10 + i*20,
                        "%s: %.2f%%" % (class_names[i], probs[i]*100),
                        bbox={'facecolor':'gray', 'alpha':0.3, 'pad':5})
                else:
                    plt.text(
                        image.shape[1] + 10, 
                        10 + i*20,
                        "%s: %.2f%%" % (class_names[i], probs[i]*100))
            plt.axis('off')
            
        if class_names is not None:
            # probs, [top, num_classes]
            probs = tf.squeeze(tf.nn.softmax(logits))
            # probs corresponding to ensemble_class_id
            idxes = tf.where(tf.argmax(probs, axis=1) == ensemble_class_id)[:,0]
            probs = tf.gather(probs, idxes, axis=0)
            # mean probs over ensembled class ids, [num_classes, ]
            mean_probs = tf.reduce_mean(probs, axis=0).numpy()
            vis(image, class_names, mean_probs)
            plt.title(class_names[ensemble_class_id])
            plt.show()
            
        return ensemble_class_id
