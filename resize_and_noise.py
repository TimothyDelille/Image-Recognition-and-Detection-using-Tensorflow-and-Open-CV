#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 17:34:59 2019

@author: timothydelille
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import glob

def noisy(image):
    """
    image : ndarray
    Input image data. Will be converted to float.
    mode : str

    Gaussian-distributed additive noise.
    """
    row,col,ch= image.shape
    mean = 0
    var = 1000
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy

if __name__=="__main__":
    cpt=0
    scale_percent = 17 # percent of original size
    #images dans le meme repertoire que le fichier python
    for filename in glob.glob('*.JPG'): #assuming jpg
        image=cv2.imread(filename,1) #1=on ne prend pas en compte la transparence (seulement 3 channels pour tensorflow)
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image_resize = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        image_noisy = noisy(image_resize)
        cv2.imwrite('images traitees/iphone_'+str(cpt)+'_noisy.JPG',image_noisy)
        cv2.imwrite('images traitees/iphone_'+str(cpt)+'_resize.JPG',image_resize)
        cpt+=1
        