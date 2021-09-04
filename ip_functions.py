import os
import skimage
from skimage import io

import matplotlib.pyplot as plt
import numpy as np

def ndim2flat(img):
    np_img = np.array(img)
    
    if len(np.shape(np_img)) == 2:
        flat = img.reshape((np.shape(np_img)[0]*np.shape(np_img)[1],))
    elif len(np.shape(np_img)) == 3:
        flat = img.reshape((np.shape(np_img)[0]*np.shape(np_img)[1]*np.shape(np_img)[2]))        
    return flat

def flatten(img):
  np_img = np.array(img)
  cnt = 0
  y_len = np.shape(np_img)[0];
  x_len = np.shape(np_img)[1];
  flat_arr = np.zeros((y_len*x_len))
  
  # Create 1 dimensional array
  for yy in range(y_len-1):
      for xx in range(x_len-1):
          flat_arr[cnt] = np_img[yy,xx]        
          cnt += 1
  return flat_arr

def flat2hist(flat_img, num_bins, plot_enable):

  if plot_enable == 1:
      print('Plot enabled')
  
  img = np.array(flat_img)
  img_max = np.amax(img)
  print('max: ', img_max)
  img_min = np.amin(img)
  print('min: ', img_min)  
  img_len = np.shape(img)[0]
  epsilon = float((img_max - img_min)/num_bins)
  bins = np.arange(img_min, img_max, epsilon)
  #bins = np.arange(img_min, img_max)  
  bins_count = np.zeros((len(bins)))
  bins_len = len(bins)

  # Create 256 bins for histogram        
  for ii in range(img_len-1):
      for bin in range(bins_len-1):
          if img[ii] >= bins[bin] and img[ii] < bins[bin] + epsilon:
              bins_count[bin] += 1
  return np.array([bins, bins_count])

def color2grey(img):
    b = [.3, .6, .1]
    return np.dot(img[...,:3], b)
              


  




