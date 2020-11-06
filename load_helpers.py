#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image




# Helper functions

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop(im, patch_dimensions):
    # We treat square patches
    list_patches = []
    # We treat square images
    img_dimensions = im.shape[0]
    #####PB PADDING SIZE
    padding_size = int((img_dimensions - patch_dimensions)/2)
    is_2d = len(im.shape) < 3
    #### PB IN RANGE BOUCLES FOR
    for i in range(0,img_dimensions,patch_dimensions):
        for j in range(0,img_dimensions,patch_dimensions):
            # pad the data before treating it
            if is_2d:
                im1 = np.pad(im, ((padding_size, padding_size), (padding_size, padding_size)), 'reflect')
                ###### PB HEREEEEEEEEE
                im_patch = im1[j:j+patch_dimensions, i:i+patch_dimensions]
            else:
                im1 = np.pad(im, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), 'reflect')
                im_patch = im1[j:j+patch_dimensions, i:i+patch_dimensions, :]
            list_patches.append(im_patch)
    return list_patches

def value_to_class(v,foreground_threshold=0.5):
    #### DES TRUCS A CHANGER !!!!!!!!!!!!
    df = np.sum(v)
    if df > foreground_threshold : return 1
    else : return 0
