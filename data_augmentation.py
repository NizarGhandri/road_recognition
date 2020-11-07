#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.image as mpimg
import numpy as np
import os,sys
import random
from PIL import Image
from skimage.util import random_noise
from keras.preprocessing.image import array_to_img
from PIL import Image, ImageEnhance
from keras.preprocessing.image import img_to_array

def rotate_images(img1 , img2 ):
    """
    randomly chooses to rotate images but angles of 90° or 180°
    same rotation applies to both satellite image and its ground truth
    """
    img1_copy= img1.copy()
    img2_copy= img2.copy()
    k= random.randint(0, 2)
    img1_copy=np.rot90(img1_copy ,k)
    img2_copy=np.rot90(img2_copy,k)
          
    return img1_copy, img2_copy

def add_noise(img, ratio): 
    """
    Add salt and pepper noise randomly
    """
    img_copy= img.copy()
    return random_noise(img_copy, mode='s&p', amount=ratio, seed=None, clip=True)

def augment_dataset(imgs,gt_imgs,image_directory,gt_directory,max_number_iterations):
    """
    main function : made to add new images to the dataset
    number_iterations : number of time we do the work of adding images
    What it does (randomly):
    i)rotate images
    ii)flip images
    iii)add noise to the images
    iv)change brightness of the images
    v)zooms in the images
    vi) height and width shift
    """

    imgs_array = np.array([img_to_array(img) for img in imgs])
    gt_imgs_array = np.array([img_to_array(img) for img in gt_imgs])

    rotated = [rotate_images(img[0] , img[1]) for img in  zip(imgs_array, gt_imgs_array)]
    unziped_rotated= list(zip(*rotated))
    imgs_array_rotated= np.array(unziped_rotated[0])
    gt_imgs_array_rotated= np.array(unziped_rotated[1]) 

    #noise :
    imgs_array_rotated_augmented = add_noise(imgs_array_rotated, 0.02)
    
    #setup dicts : slightly different for images and their groundtruth : 
    #no change of brightness in groundtruth images
    data_gen_args_img = dict(
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.2,
            rotation_range=None,     
            horizontal_flip=True,
            vertical_flip=True,
            data_format=None,
            brightness_range= [0.5,1.5])

    data_gen_args_gt = dict(
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.2,
            rotation_range=None,     
            horizontal_flip=True,
            vertical_flip=True,
            data_format=None,
            brightness_range= None)

    image_datagen = ImageDataGenerator(**data_gen_args_img , fill_mode ='reflect')
    gt_datagen = ImageDataGenerator(**data_gen_args_gt, fill_mode ='reflect')

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen.fit(imgs_array_rotated_augmented, augment=True, seed=seed)
    gt_datagen.fit(gt_imgs_array_rotated, augment=True, seed=seed)

    image_generator = image_datagen.flow(imgs_array_rotated_augmented,  batch_size=12,  
                              save_to_dir=image_directory, 
                              save_prefix='aug_img', 
                              save_format='png', seed=seed)

    gt_generator = gt_datagen.flow(gt_imgs_array_rotated, batch_size=12,  
                              save_to_dir=gt_directory, 
                              save_prefix='aug_gt', 
                              save_format='png', seed=seed)
    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, gt_generator)
    
    i=0
    for batch in train_generator:

        i += 1
        if i > max_number_iterations:
            break
    
    

