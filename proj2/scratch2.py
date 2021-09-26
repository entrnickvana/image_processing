import pandas as pd
from skimage import io
import numpy as np
import random
import skimage as ski
import csv
import os
import matplotlib.pyplot as plt
import code


def noiseAddtion(image,noiseIndicator):
    # This function gets an image of size m x n. Range of values is (0,1).
    # noiseIndicator is type of noise that needs to be added to the image
    # noiseIndicator == 0 indicates an addition of gaussian noise with mean 0 and var 0.08
    # noiseIndicator == 1 indicates an addtion of Salt and Pepper noise with intensity variation of 0.08
    # noiseIndicator == 2 indicates an addition of poisson noise
    # noiseIndicator == 3 indicates an addition of Speckle noise of mean 0 and var 0.05
    
    ## This function should return a noisy version of the input image
    
    ##  ***************** Your Code starts here ***************** ##

    if( noiseIndicator == 0 ):
      noisy = ski.util.random_noise(image, mode='gaussian', seed=None, mean=0, var=0.08)
    elif noiseIndicator == 1 :
      noisy = ski.util.random_noise(image, mode='s&p', seed=None, amount=0.08)
    elif noiseIndicator == 2 :
      noisy = ski.util.random_noise(image, mode='poisson', seed=None)
    elif noiseIndicator == 3 :
      noisy = ski.util.random_noise(image, mode='speckle', seed=None, mean=0, var=0.05)
    else:
        print('Something went wrong with NoiseIndicator\n')
        
    ## ***************** Your Code ends here ***************** ##
    return noisy

img1 = './proj2/Project2_Files-Scripts/Project2_Files-Scripts/utils/GrayScale/0.png'

#im = io.imread('Project2_Files-Scripts/Project2_Files-Scripts/utils/GrayScale/0.png')
im = io.imread(img1)
#im_noise = noiseAddtion(im, 0)

plt.subplot(321)
plt.imshow(im, cmap = 'gray')
plt.subplot(322)
plt.imshow(noiseAddtion(im, 0), cmap = 'gray')
plt.subplot(323)
plt.imshow(noiseAddtion(im, 1), cmap = 'gray')
plt.subplot(324)
plt.imshow(noiseAddtion(im, 2), cmap = 'gray')
plt.subplot(325)
plt.imshow(noiseAddtion(im, 3), cmap = 'gray')
plt.show()

