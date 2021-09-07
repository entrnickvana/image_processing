
import code
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

def thresh(img, upper, lower, center, val):
  cnt = 0
  new_img = np.array(img)
  print('shape orig: ', np.shape(img), 'new shape: ', np.shape(new_img))
  y_len = np.shape(img)[0];
  x_len = np.shape(img)[1];
  
  for yy in range(y_len-1):
      for xx in range(x_len-1):
          if img[yy,xx] > center - lower and img[yy,xx] <= center + upper:
            new_img[yy,xx] = val;
        
  return new_img

def thresh_div(img, val):
  cnt = 0
  new_img = np.array(img)
  y_len = np.shape(img)[0];
  x_len = np.shape(img)[1];
  mmax = np.amax(img)
  mmin = np.amin(img)
  img_span = mmax-mmin
  mmid = mmin + (img_span/2)
  
  for yy in range(y_len-1):
      for xx in range(x_len-1):
          if img[yy,xx] < mmid:
            new_img[yy,xx] = val
  return new_img            


def thresh_div(img, thresh_val, new_val):
  cnt = 0
  new_img = np.array(img)
  y_len = np.shape(img)[0];
  x_len = np.shape(img)[1];
  mmax = np.amax(img)
  mmin = np.amin(img)
  img_span = mmax-mmin
  mmid = mmin + (img_span/2)
  
  for yy in range(y_len-1):
      for xx in range(x_len-1):
          if img[yy,xx] < thresh_val:
            new_img[yy,xx] = new_val
  return new_img            


def flat2hist(flat_img, num_bins, plot_enable):

  if plot_enable == 1:
      print('Plot enabled')
  
  img = np.array(flat_img)
  img_max = np.amax(img)
  img_min = np.amin(img)
  img_len = np.shape(img)[0]
  epsilon = float((img_max - img_min)/num_bins)
  bins = np.arange(img_min, img_max, epsilon)
  bins_count = np.zeros((len(bins)))
  bins_len = len(bins)

  # Create 256 bins for histogram        
  for ii in range(img_len-1):
      for bin in range(bins_len-1):
          if img[ii] >= bins[bin] and img[ii] < bins[bin] + epsilon:
              bins_count[bin] += 1
              break
  return np.array([bins, bins_count])

def color2grey(img):
    b = [.3, .6, .1]
    return np.dot(img[...,:3], b)

def comp_label_gray_debug(img, x, y, targ_val, tolerance, nbrhd_width):
    count = 0
    half = int(nbrhd_width/2)
    print('y: ', y)
    print('x: ', x)        
    print('nbrhd: ', nbrhd_width)
    print('half: ', half)
    print('y+half', y+half, 'y-half', y-half)
    print('x+half', x+half, 'x-half', x-half)
    print('y slice: ', img[y-half:y+half+1, x-half:x+half+1])
    new_img = np.array(img[y-half:y+half+1, x-half:x+half+1])
    plt.imshow(new_img, cmap='gray')
    plt.show()
    

def comp_label_gray(img, y, x, targ_val, tolerance, nbrhd_width):
    count = 0
    half = int(nbrhd_width/2)
    print('half', half)
    y_len = np.shape(img)[0];
    print('y_len', y_len)
    x_len = np.shape(img)[1];
    print('x_len', x_len)
    x_neg = x-half
    print('x_neg', x_neg)
    y_neg = y-half
    print('y_neg', y_neg)
    y_over = y_len-(y+half)-1
    print('y_over', y_over)
    x_over = x_len-(x+half)-1
    print('x_over', x_over)    
    

    ## Top left --
    if (x_neg < 0) and (y_neg < 0):
      print('11111')
      new_img = np.array(img[0:y+half+1, 0:x+half+1])        
    ## Top right --
    elif (x_over < 0) and (y_neg < 0):
      print('22222')        
      new_img = np.array(img[0:y+half+1, x-half:x_len])                
    ## bottom left --
    elif (x_neg < 0) and (y_over < 0):
      print('333333')                
      new_img = np.array(img[y-half:y_len, 0:x+half+1])        
    ## bottom right
    elif (x_over < 0) and (y_over < 0):
      print('4444444')                        
      new_img = np.array(img[y-half:y_len, x-half:x_len]) 
    ## Top --
    elif (y_neg < 0):
      print('5555555')                                
      new_img = np.array(img[0:y+half+1, x-half:x+half+1])
    ## Bottom --
    elif (y_over < 0):
      print('66666666')                                        
      new_img = np.array(img[y-half:y_len, x-half:x+half+1])        
    ## left --
    elif (x_neg < 0):
      print('77777777')                                                
      new_img = np.array(img[y-half:y+half+1, 0:x+half+1])        
    ## right --
    elif (x_over < 0):
      print('88888888')                                                        
      new_img = np.array(img[y-half:y+half+1, x-half:x_len])
    else:
      print('99999999')                                                                
      new_img = np.array(img[y-half:y+half+1, x-half:x+half+1])
      
    plt.imshow(new_img, cmap='gray')
    plt.show()
    

            

    


  




