
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

def thresh_uniform(img, num_bins):
  new_img = np.array(img)
  mmax = np.amax(img)
  mmin = np.amin(img)
  sspan = mmax-mmin
  seg = sspan/num_bins
  half = seg/2
  
  for ii in range(num_bins):
      new_img = thresh(new_img, half, half, (ii*seg)+half, (ii*seg)+half)
      
  return new_img

def thresh_uniform_ends(img, num_bins):
  new_img = np.array(img)
  mmax = np.amax(img)
  mmin = np.amin(img)
  sspan = mmax-mmin
  seg = sspan/num_bins
  half = seg/2
  
  for ii in range(num_bins):
      if ii == 0:
        new_img = thresh(new_img, half, 0, mmin, mmin)
        continue
      if ii == num_bins-1:
        new_img = thresh(new_img, 0, half, mmax, mmax)
        continue
      
      new_img = thresh(new_img, half, half, (ii*seg)+half, (ii*seg)+half)
      
  return new_img

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


def majority(nbrhd, targ_val, tolerance):
    #plt.figure(3)
    #plt.imshow(nbrhd, cmap='gray')
    #plt.show()
    y_len = np.shape(nbrhd)[0];
    x_len = np.shape(nbrhd)[1];
    print('y_len: ', y_len, '\tx_len', x_len )    
    sum1 = 0
    count = 0
    is_majority = 0
    for yy in range(y_len):
        for xx in range(x_len):
            if (nbrhd[yy, xx] >= targ_val - tolerance) and (nbrhd[yy, xx] < targ_val + tolerance):
              sum1+= 1
            print('sum: ', sum1, '\tcount: ', count, '\tcurr val: ', nbrhd[yy, xx], 'target: ', targ_val -tolerance, '  ',targ_val + tolerance)
            count+=1

    print('result', (sum1/count))
    if ((sum1/count) >= 0.5):
        is_majority = 1

    print('is_majority: ', is_majority)
        
    return is_majority

def comp_label_gray(img, y, x, targ_val, tolerance, nbrhd_width):
    count = 0
    half = int(nbrhd_width/2)
    y_len = np.shape(img)[0];
    x_len = np.shape(img)[1];
    x_neg = x-half
    y_neg = y-half
    y_over = y_len-(y+half)-1
    x_over = x_len-(x+half)-1

    ## Top left --
    if (x_neg < 0) and (y_neg < 0):
      new_img = np.array(img[0:y+half+1, 0:x+half+1])
    ## Top right --
    elif (x_over < 0) and (y_neg < 0):
      new_img = np.array(img[0:y+half+1, x-half:x_len])                
    ## bottom left --
    elif (x_neg < 0) and (y_over < 0):
      new_img = np.array(img[y-half:y_len, 0:x+half+1])        
    ## bottom right
    elif (x_over < 0) and (y_over < 0):
      new_img = np.array(img[y-half:y_len, x-half:x_len]) 
    ## Top --
    elif (y_neg < 0):
      new_img = np.array(img[0:y+half+1, x-half:x+half+1])
    ## Bottom --
    elif (y_over < 0):
      new_img = np.array(img[y-half:y_len, x-half:x+half+1])        
    ## left --
    elif (x_neg < 0):
      new_img = np.array(img[y-half:y+half+1, 0:x+half+1])        
    ## right --
    elif (x_over < 0):
      new_img = np.array(img[y-half:y+half+1, x-half:x_len])
    else:
      new_img = np.array(img[y-half:y+half+1, x-half:x+half+1])

    #print('comp label gray: ', np.shape(new_img))
    #plt.imshow(new_img, cmap='gray')
    #plt.show()

    return majority(new_img, targ_val, tolerance)

    
def c_label(img, targ_val,tolerance, nbrhd_width):

    if(nbrhd_width % 2 == 0):
        print('nbrhd_width: ', nbrhd_width, '  must be an odd number')
        return 0

    new_img = np.array(img)
    y_len = np.shape(img)[0];
    x_len = np.shape(img)[1];
    for yy in range(y_len):
        for xx in range(x_len):
            if(comp_label_gray(img, yy, xx, targ_val, tolerance, nbrhd_width) > 0):
                new_img[yy, xx] = targ_val;
            #plt.imshow(new_img, cmap='gray')
            #plt.show()
            
    return new_img

def c_label_bins(img, targ_vals,tolerance, nbrhd_width):

    if(nbrhd_width % 2 == 0):
        print('nbrhd_width: ', nbrhd_width, '  must be an odd number')
        return 0

    targ_vals_len = len(targ_vals)
    new_img = np.array(img)
    y_len = np.shape(img)[0];
    x_len = np.shape(img)[1];
    for yy in range(y_len):
        for xx in range(x_len):
            for targs in range(targ_vals_len-1):
              if(comp_label_gray(img, yy, xx, targ_vals[targs], tolerance, nbrhd_width) > 0):
                new_img[yy, xx] = targ_vals[targs];
            #plt.imshow(new_img, cmap='gray')
            #plt.show()
            
    return new_img

            

    


  




