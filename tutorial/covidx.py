# -*- coding: utf-8 -*-
"""
Created on Sun May  9 10:31:49 2021

@author: Ming Ming Zhang, mmzhangist@gmail.com

Focal Loss on COVIDx Dataset
"""

import numpy as np
import tensorflow as tf
import os, time
import concurrent.futures as cf
from skimage.io import imread, imsave
from skimage.color import gray2rgb

import resnet_fpn


def move_imgs(dataset_dir, subset='train'):
    assert subset in ['train', 'test']
    if subset == 'train':
        split_txt_path = os.path.join(dataset_dir, 'train_split.txt')
    else:
        split_txt_path = os.path.join(dataset_dir, 'test_split.txt')
        
    with open(split_txt_path, 'r') as file:
        info = file.readlines()
    
    imgs_list, classes_list = [], []
    for an_info in info:
        imgs_list.append(an_info.split()[1])
        classes_list.append(an_info.split()[2])

    last_dir_path = os.path.join(dataset_dir, subset)
    classes = np.unique(np.array(classes_list))
    for class_name in classes: 
        path = os.path.join(last_dir_path, class_name)
        if not os.path.exists(path):
            os.mkdir(path)
            
    t1 = time.time()
    with cf.ThreadPoolExecutor() as executor:
        for idx in range(len(imgs_list)):
            from_file = os.path.join(last_dir_path, imgs_list[idx])
            to_file = os.path.join(last_dir_path, classes_list[idx], imgs_list[idx])
            executor.submit(os.rename, from_file, to_file)
    t2 = time.time()
    print('Processed %s %.1f seconds' % (subset, t2-t1))

    
def convert_rgb(img_path):
    #print(img_path)
    image = imread(img_path)
    if image.ndim != 3:
        image = gray2rgb(image)
        imsave(img_path, image)
    elif image.shape[-1] == 4:
        image = image[..., :3]
        imsave(img_path, image)
 
    
def covert_rgb_dir(directory):
    error_count = 0
    classes = os.listdir(directory)
    t1 = time.time()
    for class_name in classes:
        imgs_dir = os.path.join(directory, class_name)
        imgs_list = os.listdir(imgs_dir) 
        
        for idx in range(len(imgs_list)):
            try:
                img_name = imgs_list[idx]
                img_path = os.path.join(imgs_dir, img_name)
                convert_rgb(img_path)
            except:
                print('Error in processing image %s' % (img_path))
                error_count += 1
                if error_count > 5:
                    raise
   
        # with cf.ThreadPoolExecutor() as executor:
        #     for idx in range(len(imgs_list)):
        #         img_name = imgs_list[idx]
        #         img_path = os.path.join(imgs_dir, img_name)
        #         executor.submit(convert_rgb, img_path)
    t2 = time.time()
    print('Time %.1f seconds' % (t2-t1))


def load_data(dataset_dir, batch_size=32, image_size=(256,256)):
    for subset in ['train', 'test']:
        subset_dir = os.path.join(dataset_dir, subset)
        if subset == 'train':
            train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                directory=subset_dir, 
                batch_size=batch_size, 
                image_size=image_size, 
                seed=123, 
                validation_split=0.2, 
                subset='training'
                )
            val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                directory=subset_dir, 
                batch_size=batch_size, 
                image_size=image_size, 
                seed=123, 
                validation_split=0.2, 
                subset='validation'
                )
        else:
            test_ds = tf.keras.preprocessing.image_dataset_from_directory(
                directory=subset_dir, 
                batch_size=batch_size, 
                image_size=image_size
                )
    classes = train_ds.class_names
    train_ds = train_ds.cache().shuffle(1000).prefetch(
        tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.experimental.AUTOTUNE)
    return train_ds, val_ds, test_ds, classes


def tune_focal(
        train_ds, 
        val_ds, 
        classes, 
        alphas, 
        gammas, 
        image_shape=(256,256,3), 
        resnet_weights_path=None
        ):
    best_f1 = 0
    t1 = time.time()
    for alpha in alphas:
        print('alpha', alpha)
        for gamma in gammas:
            print('\n\ngamma:', gamma)
            ResFPN = resnet_fpn.ResFPN_Classifier(
                image_shape=image_shape, 
                num_classes=len(classes), 
                num_filters=256, 
                architecture='resnet50', 
                augmentation=False,
                checkpoint_path=None,
                resnet_weights_path=resnet_weights_path)

            ResFPN.train(
                train_dataset=train_ds, 
                val_dataset=val_ds, 
                params={
                    'lr':0.01, 'l2':0.1, 'epochs':5, 
                    'alpha':alpha, 'gamma':gamma},
                loss_type='focal',
                save_weights=False)
            
            top_idxes, val_acc = ResFPN.select_top(val_ds, top=3)
            ensemble_class_ids, metrics, f1 = ResFPN.predict(
                val_ds, classes, display_metrics=True)
            
            if f1 > best_f1:
                best_f1 = f1
                best_alpha = alpha
                best_gamma = gamma
    t2 = time.time()
    print('\ntuning time %.2f' %(t2-t1))
    print('\n-----best F1 %.2f @ alpha=%s, gamma=%f-----' \
          % (best_f1, str(best_alpha), best_gamma))    
    return best_f1, best_alpha, best_gamma