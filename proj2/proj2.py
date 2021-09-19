#    Project 2: Image denoising with convolution, nonlinear process, and neural networks
#    Due Sep 28 by 11:59pm Points 10
#    This project will address the problem of "denoising" images with different methods.  Besides
#    the particular tasks listed, the student will need to find data that extends
#    the datasets provides, quantify noise levels in images (relative to a noiseless "ground
#    truth"), produce sets of noisy images with known/documented noise characteristics.   For
#    all of the methods described the student should experiment with parameters and different
#    data/images and comment on the results and how it relates to the methodology

import code
import os
import skimage
from skimage import io

import matplotlib.pyplot as plt
import numpy as np
from ip_functions import *
from skimage import data, filters, color, morphology, exposure, img_as_float, img_as_ubyte
from skimage.util import img_as_ubyte
from skimage.segmentation import flood, flood_fill
from skimage.morphology import extrema
from skimage.exposure import histogram

# Image files
chang  = io.imread('../my_images/chang.tif') # Grey Image
church =    io.imread('../my_images/church.tif')
crowd = io.imread('../my_images/crowd.tif')
f18 = io.imread('../my_images/f18_super_hornet.png')
hound = io.imread('../my_images/houndog1.png')
iceland = io.imread('../my_images/iceland.png')
iceland2 = io.imread('../my_images/iceland2.png')
iceland3 = io.imread('../my_images/iceland3.png')
portal = io.imread('../my_images/portal.tif') # Grey Image
shanghai_building = io.imread('../my_images/shanghai_building.jpg')
bund = io.imread('../my_images/Shanghai_Bund_009.jpg')
shanghai_street = io.imread('../my_images/shanghai-streets-11a.jpg')
shanghai = io.imread('../my_images/shanghai-streets-3.jpg')
shapes_noise = io.imread('../my_images/shapes_noise.tif')
turkey = io.imread('../my_images/turkeys.tif')
xray = io.imread('../my_images/xray.png')
face_gray = io.imread('../my_images/face_gray.jpg')
ducati = io.imread('../my_images/Ducati.png')
camera = io.imread('../my_images/camera.png')

def prob(CDF, idx):
    idx = int(len(CDF)*idx)
    return CDF[idx]

def cdf1(pdf):
    cdf = np.zeros((len(pdf),))
    for ii in range(len(pdf)):
      cdf[ii] = sum(pdf[0:ii])
    return  cdf

# 1) Build and experiment with several different linear filters (also different sizes)
#    using correlation/convolution.  Quantify (e.g. using MSE) their effectiveness (and compare
#    quantitatively and qualitatively) with different levels of noise, types of noise, and images.

camera_gray = color2grey(camera)
#plt.imshow(camera)
#plt.show()
#
#plt.imshow(camera_gray, cmap='gray')
#plt.show()

y_len = np.shape(camera_gray)[0]
x_len = np.shape(camera_gray)[1]
histogram, bin_edges = np.histogram(img_as_float(camera_gray), 512)
plt.plot(histogram)
plt.show()

#np.random.gamma()


#  2) Experiment with 3 different nonlinear denoising methods, such as bilateral filtering.
#     If you use a method that was not discussed in class, explain how it works.  As in 1)
#     of this assignment experiment, quantify, and show results for different types and levels
#     of noise, parameters, etc. By extending the given code (here for .py files  Download here for .py filesand here
#     for jupyter notebooks  Download here for jupyter notebooks), build and train a neural
#     network with only linear activation functions.  Experiment with several different architectural choices
#     including number of layers, sizes of kernels, etc.  Quantify the behavior, demonstrate
#     that it trains, compare results with above on different images, levels/types of noise, etc.


#  3) By extending the given code, build and train a neural
#     network with nonlinear activation functions.  Experiment with several different architectural choices including number
#     of layers, sizes of kernels, etc.  Quantify the behavior, demonstrate that it trains, compare results with above
#     on different images, levels/types of noise, etc.

#  Regarding the NN part of the project, there are suggestions/hints of architectures to try in the example code,
#  however you should also try some of your own ideas for architectures and document how they performed.     

#  For libraries, please use only the follow (thats all you need):  
#     skimage , numpy, matplotlib, pandas,  pytorch, random, argparse.  
#     Please do not use open-cv, it makes it harder to grade/evaluate your work.
