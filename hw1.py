
import code
import os
import skimage
from skimage import io

import matplotlib.pyplot as plt
import numpy as np
from ip_functions import *
from skimage import data, filters, color, morphology, exposure, img_as_float
from skimage.util import img_as_ubyte
from skimage.segmentation import flood, flood_fill
from skimage.morphology import extrema

plt.rcParams['font.size'] = 8


## Part 1:
#    Preliminaries:  You will need to be able to read images from a file (e.g. jpg or png), convert, as needed,
#    to greyscale (make a function for this, using numpy "dot" command),
#    display images (with a greyscale colormap), save images (for use in your report).
#




#np.random.seed(1)
#a_rand = np.random.rand(12,12)*16
#a_rand = a_rand.astype(np.uint8)
#a_step = np.arange(12*12).reshape((12,12)).astype(np.uint8)
#a = a_rand + a_step
#a_thresh = np.copy(a)
#a_thresh[a_thresh > 100] = 0
##thresh = 100
##a_thresh[a_thresh > thresh] = 1
##a_thresh[a_thresh <= thresh] = 0
#
#plt.subplot(221)
#plt.imshow(a, cmap='gray')
#
#plt.subplot(222)
#plt.imshow(a_thresh, cmap='gray')
#plt.show()
#exit()
#gap = np.array([[0, 0, 0, 0, 0, 0, 0],                                
#                [0, 1, 1, 1, 1, 0, 0],
#                [0, 1, 0, 0, 1, 0, 0],
#                [0, 1, 0, 0, 0, 0, 0],
#                [0, 1, 0, 0, 1, 0, 0],
#                [0, 1, 1, 1, 1, 0, 0],                
#                [0, 0, 0, 0, 0, 0, 0]                                
#                ])
#
#np.random.seed(1)
#m_rnd = np.random.rand(7,7)*10
#
#
#gap = np.array([[0, 0, 0, 0, 0, 0, 0],                                
#                [0, 1, 1, 1, 1, 0, 0],
#                [0, 1, 0, 0, 1, 0, 0],
#                [0, 1, 0, 1, 1, 0, 0],
#                [0, 1, 0, 1, 1, 0, 0],
#                [0, 1, 1, 1, 1, 0, 0],                
#                [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
#
#gap = np.multiply(gap*64, m_rnd).astype(np.uint8)
#
#gap_holes = morphology.remove_small_holes(gap, 2)

#plt.subplot(221)
#plt.imshow(gap, cmap='gray')
#
#plt.subplot(222)
#plt.imshow(gap_holes, cmap='gray')
#plt.show()


#h1 = np.arange(64).reshape((8,8))*4
#print(h1)
#plt.subplot(221)
#plt.imshow(h1, cmap='gray')
#
#h1cpy = np.copy(h1)
#h1cpy[h1 < 64] = 1
#plt.subplot(222)
#plt.imshow(h1cpy, cmap='gray')
#plt.show()


#Provided images and some of my own images
chang = io.imread('my_images/chang.tif') # Grey Image
#img = io.imread('my_images/church.tif')
#img = io.imread('my_images/crowd.tif')
f18_img = io.imread('my_images/f18_super_hornet.png')
#img = io.imread('my_images/houndog1.png')
iceland2 = io.imread('my_images/iceland2.png')
#img = io.imread('my_images/iceland3.png')
iceland = io.imread('my_images/iceland.png')
img4 = io.imread('my_images/portal.tif') # Grey Image
#img = io.imread('my_images/shanghai_building.jpg')
#img = io.imread('my_images/Shanghai_Bund_009.jpg')
#img = io.imread('my_images/shanghai-streets-11a.jpg')
shanghai = io.imread('my_images/shanghai-streets-3.jpg')
shapes_noise = io.imread('my_images/shapes_noise.tif')
turkey = io.imread('my_images/turkeys.tif')
#img = io.imread('my_images/xray.png')
face_gray = io.imread('my_images/face_gray.jpg')

##
# Histogram equalization:  Perform histogram equalization on a
# selection of images, show the histograms before and after
# equalization, and comment on the visual results.
# Perform adaptive and/or local histogram equalization on a
# variety of images.  Identify all important parameters,
# experiment with (vary) those parameters, and report results on
# selection of images of different types (photos, medical, etc.).


# Notes on imshow
# input may be RGB(A) data or 2D scalar data which will be rendered as psuedocolor image
# Colormapping parameters (cmap = 'gray', vmin = 0, vmax = 255)
# Pixel info: The number of pixels used to rendan an image is set by the Axes size and the dpi of the figure
# Supported formats of array X:
#   1. (M,N): an image with scalar data the values are mapped to colors using normalization and a colormap
#   2. (M,N, 3): an image with RGB values ([0,2] for float, [0,255] for int)
#   3. (M,N, 4): an image with RGBA -> {Red, Green, Blue, Alpha} values ([0,2] for float, [0,255] for int)
#       a.  What is Alpha? What is PIL?
#             Alpha: Level of Transparency and mixing of colors
#             PIL: Python Imaging library
#

comp_eq(chang, 512, 'Chang')

exit()

## https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html
#def hist1(image, axes, bins=256):
#    """Plot an image along with its histogram and cumulative histogram.
#
#    """
#    image = img_as_float(image)
#    ax_img, ax_hist = axes
#    code.interact(local=locals())
#    ax_cdf = ax_hist.twinx()
#
#    # Display image
#    ax_img.imshow(image, cmap=plt.cm.gray)
#    ax_img.set_axis_off()
#
#    # Display histogram
#    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
#    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
#    ax_hist.set_xlabel('Pixel intensity')
#    ax_hist.set_xlim(0, 1)
#    ax_hist.set_yticks([])
#
#    # Display cumulative distribution
#    img_cdf, bins = exposure.cumulative_distribution(image, bins)
#    ax_cdf.plot(bins, img_cdf, 'r')
#    ax_cdf.set_yticks([])
#
#    return ax_img, ax_hist, ax_cdf


    

# Load an example img
chang_img = img_as_float(chang)

code.interact(local=locals())
#chang_img = chang

# Contrast Strectching
p2, p98 = np.percentile(chang_img, (0,95))
chang_rescale = exposure.rescale_intensity(chang, in_range=(p2,p98))
  
# Equalization
chang_eq = exposure.equalize_hist(chang)

# Adaptive Equalization
chang_adapteq = exposure.equalize_adapthist(chang, clip_limit=0.03)

chang_hist = flat2hist(flatten(chang), 512, 0)
plt.bar(chang_hist[0], chang_hist[1], color = 'blue')
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('chang Histogram')
plt.show()

# Display results
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2, 4), dtype=object)
axes[0, 0] = fig.add_subplot(2, 4, 1)
#for i in range(1, 4):
#    axes[0, i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
#for i in range(0, 4):
#    axes[1, i] = fig.add_subplot(2, 4, 5+i)

ax_img, ax_hist, ax_cdf = hist1(chang_rescale, axes[:, 1])
ax_img.set_title('Contrast stretching')

    

exit()



plt.subplot(221)
plt.imshow(turkey, cmap='gray')

plt.subplot(222)
#turkey_hist = flat2hist(flatten(turkey), 512, 0)
#plt.bar(turkey_hist[0], turkey_hist[1], color = 'blue')
#plt.xlabel('Bin')
#plt.ylabel('Count')
#plt.title('turkey Histogram')

turkey_thresh = np.copy(turkey)
turkey_thresh[turkey_thresh > 60] = 0
plt.imshow(turkey_thresh, cmap='gray')

plt.subplot(223)
turkey_holes = morphology.remove_small_holes(turkey_thresh, 1024*4)
plt.imshow(turkey_holes, cmap='gray')
plt.show()

code.interact(local=locals())

exit()

## Isolate hair in iceland
d1 = [210, 240]
d2 = [185, 205]
d3 = [170, 184]
d3 = [100, 160]
ff1 = ()
hair = [80, 100]
bodies = [110, 200]
steam = []

iceland_gray = color2grey(iceland)
iceland_hair = thresh(iceland_gray, d1[1]-d1[0], 0, d1[0], (d1[1]-d1[0]/2))
iceland_hair = thresh(iceland_hair, d2[1]-d1[0], 0, d2[0], (d2[1]-d2[0]/2))
iceland_hair = thresh(iceland_hair, d3[1]-d3[0], 0, d3[0], (d3[1]-d3[0]/2))
iceland_hole = morphology.remove_small_holes(iceland_hair.astype(int), 8)


iceland_flood = flood_fill(iceland_hair, (100,100), 255, tolerance=50)

iceland_flat = flatten(iceland_gray)
#iceland_hist = flat2hist(iceland_flat, 512, 0)
#plt.subplot(231)
#plt.bar(iceland_hist[0], iceland_hist[1], color = 'blue')
#plt.xlabel('Bin')
#plt.ylabel('Count')
#plt.title('shapes Histogram')

plt.subplot(232)
plt.imshow(iceland)

plt.subplot(233)
plt.imshow(iceland_hair, cmap='gray')

plt.subplot(234)
plt.imshow(iceland_hole, cmap='gray')

plt.subplot(235)
plt.imshow(iceland_flood, cmap='gray')

plt.subplot(236)
iceland_hair_flat = flatten(iceland_hair)
iceland_hist_hair = flat2hist(iceland_hair_flat, 512, 0)
plt.bar(iceland_hist_hair[0], iceland_hist_hair[1], color = 'blue')
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('shapes Histogram')
plt.show()

exit()

##flood fill









shapes_noise_flat = flatten(shapes_noise)
shapes_hist = flat2hist(shapes_noise_flat, 512, 0)
plt.bar(shapes_hist[0], shapes_hist[1], color = 'blue')
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('shapes Histogram')
plt.show()



#print('Note that images 1, 4 are grayscale images, this is how they are rendered in color, interesting')
#plt.figure(1)
#plt.subplot(221)
#plt.imshow(img1)
#plt.subplot(222)
#plt.imshow(img2)
#plt.subplot(223)
#plt.imshow(img3)
#plt.subplot(224)
#plt.imshow(img4)
#plt.show()


# Use dot product on weighted intensities to create grayscale image, code taken from TA in class
# (See my library of functions in ip_functions.py)

#plt.figure(2)
img_gray1 = img1
#plt.subplot(221)
#plt.imshow(img_gray1, cmap='gray')

#img_gray2 = color2grey(img2)
#plt.subplot(222)
#plt.imshow(img_gray2, cmap='gray')

img_gray4 = img4
#plt.subplot(224)
#plt.imshow(img_gray4, cmap='gray')
#plt.show()

## Part 2:
#    Build a histogram:  Write a function (from scratch, using iterators in numpy) that takes a greyscale
#    image/array and returns a 2D array where the first column entries are the histogram bin
#    values (start values) and the second column are the bin counts.  Display the resulting histogram using a
#    bar chart from matplotlib.   Display histograms for a couple of different images and
#    describe how they relate to what you see in the image (e.g. what regions/objects are
#    what part of the histogram).  Thresholding (below) can help with this.

#flat1 = flatten(img_gray1)
#img_hist = flat2hist(flat1, 1024, 0)
#
#plt.bar(img_hist[0], img_hist[1], color = 'green')
#plt.xlabel('Bin')
#plt.ylabel('Count')
#plt.title('Chang Image Histogram')
#plt.show()
#
#
#flat2 = flatten(img_gray2)
#img_hist2 = flat2hist(flat2, 1024, 0)
#
#plt.bar(img_hist2[0], img_hist2[1], color = 'green')
#plt.xlabel('Bin')
#plt.ylabel('Count')
#plt.title('f18 Image Histogram')
#
#flat3 = flatten(img_gray3)
#img_hist3 = flat2hist(flat3, 1024, 0)
#
#plt.bar(img_hist3[0], img_hist3[1], color = 'green')
#plt.xlabel('Bin')
#plt.ylabel('Count')
#plt.title('Iceland Image Histogram')
#
#flat4 = flatten(img_gray4)
#img_hist4 = flat2hist(flat4, 1024, 0)
#
#plt.bar(img_hist4[0], img_hist4[1], color = 'green')
#plt.xlabel('Bin')
#plt.ylabel('Count')
#plt.title('Portal Image Histogram')


## Part 3:
#    Regions and components:  Define a function that performs double-sided (high and low) thresholding on images to define regions,
#    visualize results (and histograms) on several images.   Perform flood fill and connected component on these
#    thresholded images.  Remove connected components that are smaller than a certain size (you specify).
#    Visualize the results as a color image (different colors for different regions).

# https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_floodfill.html

#plt.subplot(221)
#toy_example = np.arange(64).reshape((8,8))
#plt.imshow(toy_example, cmap='gray')
#
#plt.subplot(222)
#toy_thresh = thresh_uniform(toy_example, 2)
#plt.imshow(toy_thresh, cmap='gray')
#
#plt.subplot(223)
#toy_thresh = thresh_uniform(toy_example, 4)
#plt.imshow(toy_thresh, cmap='gray')
#
#plt.subplot(224)
#toy_thresh = thresh_uniform(toy_example, 64)
#plt.imshow(toy_thresh, cmap='gray')
#plt.show()
#
#

#http://scipy-lectures.org/packages/scikit-image/auto_examples/plot_labels.html


#n = 12
#l = 256
#np.random.seed(1)
#im = np.zeros((l, l))
#points = l * np.random.random((2, n ** 2))
#im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
#
#im = filters.gaussian(im, sigma= l / (4. * n))
#blobs = im > 0.7 * im.mean()

plt.imshow(shapes_noise, cmap = 'gray')
plt.show()


hair = [80, 100]
bodies = [110, 200]
shapes_noise_flat = flatten(shapes_noise)
shapes_hist = flat2hist(shapes_noise_flat, 512, 0)
plt.bar(shapes_hist[0], shapes_hist[1], color = 'blue')
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('shapes Histogram')
plt.show()

background_shape = [30, 90]
shapes1 = [130, 220]
shapes_thresh = thresh(shapes_noise, shapes1[1] - shapes1[0],0, shapes1[0], (shapes1[1]-shapes1[0])/2)
shapes_thresh = thresh(shapes_thresh, background_shape[1] - background_shape[0],0, background_shape[0], (background_shape[1]-background_shape[0]/2))
shapes_noise_flat = flatten(shapes_thresh)
shapes_hist_thresh = flat2hist(shapes_noise_flat, 512, 0)
plt.bar(shapes_hist_thresh[0], shapes_hist_thresh[1], color = 'green')
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('shapes thresh Histogram')
plt.show()

plt.imshow(shapes_thresh, cmap='gray')
plt.show()

## flood fill
plt.subplot(221)
plt.imshow(shapes_noise, cmap='gray')
plt.show()

shapes_flood = flood_fill(shapes_thresh, (2,2), 0, tolerance=50)
shapes_flood = flood_fill(shapes_thresh, (80, 80), (shapes1[1]-shapes1[0])/2, tolerance=30)
shapes_flood = flood_fill(shapes_thresh, (250, 150), (shapes1[1]-shapes1[0])/2, tolerance=30)
shapes_flood = flood_fill(shapes_thresh, (125, 250), (shapes1[1]-shapes1[0])/2, tolerance=30)
plt.subplot(222)
plt.imshow(shapes_flood, cmap='gray')
plt.show()

 

exit()

turkey_flat = flatten(turkey)
turkey_hist = flat2hist(turkey_flat, 512, 0)
xtrma = extrema.local_maximum()


turkey_flat = flatten(turkey)
turkey_hist = flat2hist(turkey_flat, 512, 0)
xtrma = extrema.local_maximum()


plt.bar(turkey_hist[0], turkey_hist[1], color = 'blue')
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('turkey Histogram')
plt.show()

exit()

f18_gray = color2grey(f18_img)
iceland_gray = color2grey(iceland)
iceland_flat = flatten(iceland_gray)
iceland_hist = flat2hist(iceland_flat, 512, 0)
plt.bar(iceland_hist[0], iceland_hist[1], color = 'blue')
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('Iceland Histogram')
plt.show()


iceland_hair = thresh(iceland_gray, 20, 0, 100, (hair[0]-hair[1]/2))
iceland_hair = thresh(iceland_hair, 20, 0, 100, (bodies[0]-bodies[1]/2))
iceland_hair = thresh(iceland_hair, 54, 0, 201, 255)
plt.subplot(221)
plt.imshow(iceland_gray, cmap='gray')
plt.subplot(222)
plt.imshow(iceland_hair, cmap='gray')
plt.show()


#f18_thresh = thresh_uniform(f18_gray, 3)
#iceland_thresh = thresh_uniform(iceland_gray, 3)
#plt.imshow(iceland_thresh, cmap='gray')
#plt.show()
#
## flood fill



exit()

#
#all_labels = measure.label(blobs)
#blobs_labels = measure.label(blobs, background=0)
#code.interact(local=locals())
#
#
#plt.figure(figsize=(9, 3.5))
#plt.subplot(131)
#plt.imshow(blobs, cmap='gray')
#plt.axis('off')
#plt.subplot(132)
#plt.imshow(all_labels, cmap='nipy_spectral')
#plt.axis('off')
#plt.subplot(133)
#plt.imshow(blobs_labels, cmap='nipy_spectral')
#plt.axis('off')
#
#plt.tight_layout()
#plt.show()

exit()

f18_gray = color2grey(f18_img)
plt.imshow(f18_gray, cmap='gray')
plt.show()

f18_thresh = thresh_uniform(f18_gray, 2)
plt.imshow(f18_thresh, cmap='gray')
plt.show()

f18_fill8bit = img_as_ubyte(f18_thresh)
code.interacts(local=locals())
f18_fill = morphology.remove_small_objects(f18_thresh8bit)
plt.imshow(f18_fill, cmap='gray')
plt.show()

exit()

#hound = io.imread('my_images/houndog1.png')
#hound_gray = color2grey(hound)
#hound_thresh = thresh_uniform(hound_gray, 2)
#plt.imshow(hound_thresh, cmap='gray')
#plt.show()
#
#hound_flood = flood_fill(hound_thresh, (375,474), 0, tolerance=10)
#hound_flood = flood_fill(hound_thresh, (0, 0), 0, tolerance=10)
#hound_flood = flood_fill(hound_thresh, (470, 5), 182, tolerance=10)
#plt.imshow(hound_flood, cmap='gray')
#plt.show()
#
#shanghai_gray = color2grey(shanghai)
#
### https://iq.opengenus.org/connected-component-labeling/
#face_gray = thresh_uniform(face_gray, 2)
#num_labels, labels = cv2.connectedComponents(face_gray)
##num_labels, labels = cv2.connectedComponents(shanghai_gray)
#
#label_hue = np.uint8(179*labels/np.max(labels))
#blank_ch = 255*np.ones_like(label_hue)
#labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
#
#labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
#labeled_img[label_hue == 0] = 0
#
#plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
#plt.show()

code.interact(local=locals())
exit()

#plt.subplot(221)
#plt.imshow(hound_gray, cmap='gray')
#
#plt.subplot(222)
#hound_gray_2 = thresh_uniform(hound_gray, 2)
#plt.imshow(hound_gray_2, cmap='gray')
#
#plt.subplot(223)
#hound_gray_3 = thresh_uniform(hound_gray, 3)
#plt.imshow(hound_gray_3, cmap='gray')
#
#plt.subplot(224)
#hound_gray_4 = thresh_uniform(hound_gray, 4)
#plt.imshow(hound_gray_4, cmap='gray')
#plt.show()
#
#plt.subplot(221)
#hound_flat = flatten(hound_gray)
#hound_hist = flat2hist(hound_flat, 512, 0)
#plt.bar(hound_hist[0], hound_hist[1], color = 'blue')
#plt.xlabel('Bin')
#plt.ylabel('Count')
#plt.title('Hound Histogram')
#
#plt.subplot(222)
#hound_flat_2 = flatten(hound_gray_2)
#hound_hist_2 = flat2hist(hound_flat_2, 512, 0)
#plt.bar(hound_hist_2[0], hound_hist_2[1], color = 'blue')
#plt.xlabel('Bin')
#plt.ylabel('Count')
#plt.title('Hound Uniform Thresh 2 Histogram')
#
#plt.subplot(223)
#hound_flat_3 = flatten(hound_gray_3)
#hound_hist_3 = flat2hist(hound_flat_3, 512, 0)
#plt.bar(hound_hist_3[0], hound_hist_3[1], color = 'blue')
#plt.xlabel('Bin')
#plt.ylabel('Count')
#plt.title('Hound Uniform Thresh 3 Histogram')
#
#plt.subplot(224)
#hound_flat_4 = flatten(hound_gray_4)
#hound_hist_4 = flat2hist(hound_flat_4, 512, 0)
#plt.bar(hound_hist_4[0], hound_hist_4[1], color = 'blue')
#plt.xlabel('Bin')
#plt.ylabel('Count')
#plt.title('Hound Uniform Thresh 4 Histogram')
#plt.show()

exit()

f18 = img_gray2
plt.imshow(f18, cmap='gray')
plt.show()

f18_flat = flatten(f18)
f18_hist = flat2hist(f18_flat, 512, 0)
plt.bar(f18_hist[0], f18_hist[1], color = 'blue')
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('f18 Histogram')
plt.show()


f18_32 = thresh(f18, 64, 0, 0, 0)
plt.imshow(f18_32, cmap='gray')
plt.show()

f18_flat_32 = flatten(f18_32)
f18_hist_32 = flat2hist(f18_flat_32, 512, 0)
plt.bar(f18_hist_32[0], f18_hist_32[1], color = 'blue')
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('f18 32 Threshold Histogram')
plt.show()


f18_100 = thresh(f18, 100, 0, 0, 0)
plt.imshow(f18_100, cmap='gray')
plt.show()

f18_flat_100 = flatten(f18_100)
f18_hist_100 = flat2hist(f18_flat_100, 512, 0)
plt.bar(f18_hist_100[0], f18_hist_100[1], color = 'blue')
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('f18 100 Threshold Histogram')
plt.show()



#f18_100 = thresh(f18, 100, 0, 0, 0)
#plt.subplot(325)
#plt.imshow(f18_100, cmap='gray')

#flat_thresh = flatten(gray_thresh)
#img_hist_thresh = flat2hist(flat_thresh, 512, 0)
#
#plt.bar(img_hist_thresh[0], img_hist_thresh[1], color = 'blue')
#plt.xlabel('Bin')
#plt.ylabel('Count')
#plt.title('Histogram')

code.interact(local=locals())
exit()

## Part 4:
#    Histogram equalization:  Perform histogram equalization on a selection of images, show the histograms before and
#    after equalization, and comment on the visual results.   Perform adaptive and/or local histogram equalization
#    on a variety of images.  Identify all important parameters, experiment with (vary) those parameters, and
#    report results on selection of images of different types (photos, medical, etc.).


#img = io.imread('project1_images/project1_images/houndog1.png')
#img = io.imread('project1_images/project1_images/chang.tif')
#img = io.imread('project1_images/project1_images/church.tif')





#print('img shape: ', np.shape(img))

#plt.figure(1)
#plt.subplot(211)
#np.random.seed(2)
#plt.subplot(211)
#step_img = np.arange(1024).reshape((32,32))*4
#plt.imshow(step_img, cmap='gray')
#
#plt.subplot(212)
#thresh_img = thresh_uniform(step_img, 8)
#plt.imshow(thresh_img, cmap='gray')
#plt.show()

#rnd_img = np.random.randint(low = 0, high = 3, size = 9).reshape((3,3))*64
#rnd_img = np.random.randint(low = 0, high = 3, size = 64).reshape((8,8))*64
#rnd_img_mod = np.array(rnd_img)
#gray_img = color2grey(img)

gray_img = color2grey(img)
plt.subplot(121)
plt.imshow(gray_img, cmap='gray')

gray_thresh = thresh_uniform(gray_img, 3)
plt.subplot(122)
plt.imshow(gray_thresh, cmap='gray')
plt.show()


#code.interact(local=locals())
flat = flatten(gray_img)
img_hist = flat2hist(flat, 512, 0)

plt.subplot(211)
plt.bar(img_hist[0], img_hist[1], color = 'green')
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('Histogram')


plt.subplot(212)
flat_thresh = flatten(gray_thresh)
img_hist_thresh = flat2hist(flat_thresh, 512, 0)

plt.bar(img_hist_thresh[0], img_hist_thresh[1], color = 'blue')
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('Histogram')
plt.show()

hist_xma_indices = xma(img_hist, np.greater)


code.interact(local=locals())
#for ii in hist_xma_indices:
#    print('local extrema histogram are\n index: ', ii, '\nval: ', img_hist[0, ii])
    

