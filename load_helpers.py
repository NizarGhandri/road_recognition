#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img


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

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def value_to_class_NN(v,foreground_threshold=0.25):
    df = np.sum(np.mean(v))
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

def label_to_img(imgwidth, imgheight, w, h, prediction):
    """
    w : patch width
    h : patch height
    prediction : array that was predicted by model : we will take just first column
    """
    array_labels = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            #pq???
            l=(~prediction[idx][0] > 0.5)
            #celle du prof en commentaire
            """if prediction[idx][0] > 0.5:  # bgrd
                l = 0
            else:
                l = 1"""
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels

def make_img_overlay(img, predicted_img,PIXEL_DEPTH=255):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img
