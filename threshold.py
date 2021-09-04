
import os
import skimage
from skimage import io

import matplotlib.pyplot as plt
import numpy as np
from ip_functions import *

img = io.imread('project1_images/project1_images/houndog1.png')

gray_img = color2grey(img)
plt.imshow(gray_img, cmap='gray')
plt.show()
print('2d img shape: ', np.shape(gray_img))

flat = flatten(gray_img)
print('flat img shape: ', np.shape(flat))

img_hist = flat2hist(flat, 256, 0)
print('flat img shape: ', np.shape(img_hist))

plt.bar(img_hist[0], img_hist[1], color = 'green')
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('Histogram')
plt.show()

