
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

gray_img_mod = thresh(gray_img, 20, 20, 235, 100)
plt.imshow(gray_img_mod, cmap='gray')
plt.show()

flat = flatten(gray_img)
img_hist = flat2hist(flat, 1024, 0)

plt.bar(img_hist[0], img_hist[1], color = 'green')
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('Histogram')
plt.show()

