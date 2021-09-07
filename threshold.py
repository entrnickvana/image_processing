
import code
import os
import skimage
from skimage import io
from scipy.signal import argrelextrema as xma

import matplotlib.pyplot as plt
import numpy as np
from ip_functions import *



img = io.imread('project1_images/project1_images/houndog1.png')
#img = io.imread('project1_images/project1_images/chang.tif')
#img = io.imread('project1_images/project1_images/church.tif')

#print('img shape: ', np.shape(img))

#plt.figure(1)
#plt.subplot(211)
#np.random.seed(2)
#plt.subplot(211)
#step_img = np.arange(1024).reshape((32,32))*4
#plt.imshow(step_img, cmap='gray')
#
#plt.subplot(212)
#thresh_img = thresh_uniform(step_img, 8)
#plt.imshow(thresh_img, cmap='gray')
#plt.show()

#rnd_img = np.random.randint(low = 0, high = 3, size = 9).reshape((3,3))*64
#rnd_img = np.random.randint(low = 0, high = 3, size = 64).reshape((8,8))*64
#rnd_img_mod = np.array(rnd_img)
#gray_img = color2grey(img)

gray_img = color2grey(img)
plt.subplot(121)
plt.imshow(gray_img, cmap='gray')

gray_thresh = thresh_uniform(gray_img, 3)
plt.subplot(122)
plt.imshow(gray_thresh, cmap='gray')
plt.show()


#code.interact(local=locals())
flat = flatten(gray_img)
img_hist = flat2hist(flat, 512, 0)

plt.subplot(211)
plt.bar(img_hist[0], img_hist[1], color = 'green')
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('Histogram')


plt.subplot(212)
flat_thresh = flatten(gray_thresh)
img_hist_thresh = flat2hist(flat_thresh, 512, 0)

plt.bar(img_hist_thresh[0], img_hist_thresh[1], color = 'blue')
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('Histogram')
plt.show()

hist_xma_indices = xma(img_hist, np.greater)


code.interact(local=locals())
#for ii in hist_xma_indices:
#    print('local extrema histogram are\n index: ', ii, '\nval: ', img_hist[0, ii])
    

