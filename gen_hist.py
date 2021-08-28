
import os
import skimage
from skimage import io

import matplotlib.pyplot as plt
import numpy as np


def color2grey(img):
    b = [.3, .6, .1]
    return np.dot(img[...,:3], b)

img = io.imread('project1_images/project1_images/houndog1.png')

print(img.shape)
plt.imshow(img)
plt.title('Dog')
plt.show()
 
gray_img = color2grey(img)
print('gray image:\n',gray_img.shape)
plt.imshow(gray_img, cmap='gray')
plt.show()

cnt = 0
y_len = np.shape(gray_img)[0];
x_len = np.shape(gray_img)[1];
intense_arr = np.zeros((y_len*x_len, 1))
print('shape intense:\n', np.shape(intense_arr))

# Create 1 dimensional array
for yy in range(y_len-1):
    for xx in range(x_len-1):
        intense_arr[cnt][0] = gray_img[yy][xx]
        #print('cnt: ', cnt, '\n', intense_arr[cnt][0])
        cnt += 1

intense_len = np.shape(intense_arr)[0]
match_cnt = 0
epsilon = 1/1;
hist_256 = np.zeros((256, 1))
# Create 256 bins for histogram        
for ii in range(intense_len-1):
    for bin in range(0, 254, 1):
        if intense_arr[ii] >= bin and intense_arr[ii] < bin + 1:
            hist_256[bin] += 1

plt.plot(hist_256)
plt.show()
    



        









