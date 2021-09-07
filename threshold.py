
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

np.random.seed(2)
#rnd_img = np.random.randint(low = 0, high = 3, size = 9).reshape((3,3))*64
rnd_img = np.random.randint(low = 0, high = 3, size = 64).reshape((8,8))*64
plt.imshow(rnd_img, cmap='gray')
plt.show()
comp_label_gray(rnd_img, 7, 7, 0, 0, 3)
#code.interact(local=locals())

exit()

flat = flatten(gray_img)
img_hist = flat2hist(flat, 521, 0)

plt.bar(img_hist[0], img_hist[1], color = 'green')
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('Histogram')
plt.show()

hist_xma_indices = xma(img_hist, np.greater)

for ii in hist_xma_indices:
    print('local extrema histogram are\n index: ', ii, '\nval: ', img_hist[0, ii])
    

