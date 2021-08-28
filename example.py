
import os
import skimage
from skimage import io

import matplotlib.pyplot as plt
import numpy as np


def color2grey(img):
    b = [.3, .6, .1]
    return np.dot(img[...,:3], b)

img = io.imread('project1_images/project1_images/houndog1.png')


newpid = os.fork()
if newpid == 0:
    
  print(img.shape)
  plt.imshow(img)
  plt.title('Dog')
  plt.show()
  
else:
 
 gray_img = color2grey(img)
 print(gray_img.shape)
 plt.imshow(gray_img, cmap='gray')
 plt.show()






