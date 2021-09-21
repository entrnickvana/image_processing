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


camera_gray = color2grey(camera)

def add_gamma(img_gray, gshape, gscale):
  y_len = np.shape(img_gray)[0]
  x_len = np.shape(img_gray)[1]
  g_noise = np.random.gamma(gshape, gscale, (y_len, x_len))
  gamma_img = np.add(g_noise, img_gray)
  gamma_img[gamma_img < 0] = 0.0
  return gamma_img

def add_norm(img_gray, mu, sgm):
  y_len = np.shape(img_gray)[0]
  x_len = np.shape(img_gray)[1]
  norm_noise = np.random.normal(mu, sgm, (y_len, x_len))
  norm_img = np.add(norm_noise, img_as_float(img_gray))
  norm_img[norm_img < 0] = 0.0
  return norm_img

def norm_arr(img, mu_arr, sgm_arr):
  result_arr = []
  for ii in range(len(mu_arr)):
    result_arr.append(add_norm(img, mu_arr[ii], sgm_arr[ii]))
  return result_arr

def gamma_arr(img, gshape_arr, gscale_arr):
  result_arr = []
  for ii in range(len(mu_arr)):
    result_arr.append(add_gamma(img, gshape_arr[ii], gscale_arr[ii]))
  return result_arr

def add_hist(img, bins):
  hist = np.histogram(img_as_float(img), bins)
  return hist


# 1) Build and experiment with several different linear filters (also different sizes)
#    using correlation/convolution.  Quantify (e.g. using MSE) their effectiveness (and compare
#    quantitatively and qualitatively) with different levels of noise, types of noise, and images.


#plt.subplot(311)
#plt.imshow(camera_gray, cmap='gray')

mu = [0, 0, 0, 0, 0, 0, 0, 0, 0]
norm_noise_scale = 10
sgm = np.arange(0, norm_noise_scale*9, norm_noise_scale)
cam_noises = norm_arr(camera_gray, mu, sgm)

#fig, axis = plt.subplots(3,3)
#axis = axis.ravel()
#for ii in range(len(cam_noises)):
#  axis[ii].imshow(cam_noises[ii], cmap='gray')
#  axis[ii].set_title(str(sgm[ii]))

#plt.show()
#plt.savefig('Norm_noise_levels')

mu = [0, 0, 0, 0, 0, 0, 0, 0, 0]
norm_noise_scale = 10
sgm = np.arange(0, norm_noise_scale*9, norm_noise_scale)
cam_noises = norm_arr(camera_gray, mu, sgm)
hist_arr = []
for jj in range(len(cam_noises)):
  hist_arr.append(cam_noises[jj])
  hist_arr.append(np.histogram(cam_noises[jj], 512)[0])


fig1, axis = plt.subplots(2*3,3)
axis = axis.ravel()
for ii in range(0, len(hist_arr), 2):
  axis[ii].imshow(hist_arr[ii], cmap='gray')
  axis[ii+1].plot(hist_arr[ii+1])
  axis[ii+1].set_title('hist')    

plt.show()
plt.savefig('Norm_noise_with_hist')

plt.subplot(331)
plt.imshow(cam_noises[0], cmap='gray')
plt.subplot(332)
plt.imshow(cam_noises[2]/cam_noises[2].max(), cmap='gray')
plt.subplot(333)
plt.imshow(filters.rank.mean(cam_noises[2]/cam_noises[2].max(), np.ones((3,3))), cmap='gray')
plt.subplot(334)
plt.imshow(filters.rank.mean(cam_noises[2]/cam_noises[2].max(), np.ones((5,5))), cmap='gray')
plt.subplot(335)
plt.imshow(filters.rank.mean(cam_noises[2]/cam_noises[2].max(), np.ones((7,7))), cmap='gray')
plt.subplot(336)
plt.imshow(filters.rank.mean(cam_noises[2]/cam_noises[2].max(), np.ones((9,9))), cmap='gray')
plt.subplot(337)
plt.imshow(filters.rank.mean(cam_noises[2]/cam_noises[2].max(), np.ones((11,11))), cmap='gray')
plt.subplot(338)
plt.imshow(filters.rank.mean(cam_noises[2]/cam_noises[2].max(), np.ones((13,13))), cmap='gray')
plt.subplot(339)
plt.imshow(filters.rank.mean(cam_noises[2]/cam_noises[2].max(), np.ones((15,15))), cmap='gray')
plt.show()






exit()
cam_noise = add_norm(camera_gray, 0, .5)
cam_noise *= 1.0/cam_noise.max()
plt.subplot(312)
plt.imshow(cam_noise, cmap='gray')
plt.subplot(313)
cam_denoise = filters.rank.mean(cam_noise, np.ones((3,3)))
plt.imshow(cam_denoise, cmap='gray')
plt.show()




exit()

y_len = np.shape(camera_gray)[0]
x_len = np.shape(camera_gray)[1]

gamma_noise = np.random.gamma(1.0, 1.0, (y_len, x_len))
gamma_noise2 = 2*np.random.gamma(1.0, 1.0, (y_len, x_len))
gamma_noise4 = 4*np.random.gamma(1.0, 1.0, (y_len, x_len))
gamma_noise_2 = np.random.gamma(2.0, 2.0, (y_len, x_len))
gamma_noise_4 = np.random.gamma(4.0, 4.0, (y_len, x_len))
gamma_noise_8 = np.random.gamma(8.0, 8.0, (y_len, x_len))

gn = [gamma_noise, gamma_noise2, gamma_noise4, gamma_noise_2, gamma_noise_4, gamma_noise_8]

fig, axis = plt.subplots(2,3)
axis = axis.ravel()

for ii in range(len(gn)):
    plt.ylabel('count')
    plt.xlabel('bin')
    axis[ii].imshow(gn[ii], cmap='gray')

plt.show()


exit()

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

gamma_noise0 = np.random.gamma(2.0, 2.0, (y_len, x_len))
gamma_noise10 = 10*np.random.gamma(2.0, 2.0, (y_len, x_len))
gamma_noise0_4 = 10*np.random.gamma(4.0, 4.0, (y_len, x_len))
gh0, gh0_bin = np.histogram(img_as_float(np.add(camera_gray, gamma_noise0)), 512)
plt.figure()
plt.subplot(121)
plt.imshow(camera_gray, cmap='gray')
plt.subplot(122)
plt.plot(gh0)
plt.show()

exit()

g_hist, g_edge = np.histogram(img_as_float(gamma_noise), 512)
plt.subplot(221)
plt.imshow(gamma_noise, cmap='gray')
plt.subplot(222)
plt.plot(g_hist)
plt.show()

camera_noise = np.add(img_as_float(camera_gray), gamma_noise)

noise_hist, noise_bin_edges = np.histogram(img_as_float(camera_noise), 512)

plt.subplot(221)
plt.imshow(camera_gray, cmap='gray')
plt.subplot(222)
plt.imshow(camera_noise, cmap='gray')
plt.subplot(223)
plt.plot(histogram)
plt.subplot(224)
plt.plot(noise_hist)
plt.show()





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
