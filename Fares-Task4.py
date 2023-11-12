# -*- coding: utf-8 -*-
from skimage import io, img_as_float
import numpy as np
import cv2
from skimage.filters import median, gaussian
from skimage.util import random_noise
from scipy.signal import convolve2d


img = img_as_float(io.imread("original_image.jpg", as_gray=True))
noisy_img = random_noise(img,mode='s&p',amount=.05)
noisy_img= np.array (noisy_img* 255, dtype=np.uint8)
s = io.imsave("noisy_img.jpg",noisy_img)


mean = np.ones((3,3),np.float32)/9
gaussian = np.array([[1,2,1],[2,4,2],[1,2,1]])/16


#IMG CV
img_cv = cv2.filter2D(noisy_img, -1, mean, borderType= cv2.BORDER_CONSTANT)
s = io.imsave("img_cv.jpg",img_cv)
cv2.imshow("img", img)
cv2.imshow("noisy_img", noisy_img)
cv2.imshow("img_cv", img_cv)
cv2.waitKey (0)
cv2.destroyAllWindows()

#img scripy
noisy_img = noisy_img/255
img_scipy = convolve2d(noisy_img, gaussian, mode='same')
s = io.imsave("img_scipy.jpg",img_scipy)
cv2.imshow("img",img)
cv2.imshow("noisy_img", noisy_img)
cv2.imshow("img_scipy", img_scipy)
cv2.waitKey(0)
cv2.destroyAllWindows()

#img_gaussian
m=io.imread("noisy_img.jpg", as_gray=True)
img_gaussian = cv2.GaussianBlur(m, (3,3),0, borderType= cv2.BORDER_CONSTANT)
s = io.imsave("img_gaussian.jpg",img_gaussian)
cv2.imshow("img", img)
cv2.imshow("noisy_img", noisy_img)
cv2.imshow("img_gaussian", img_gaussian)
cv2.waitKey (0)
cv2.destroyAllWindows()

#img_median
p = img_as_float(io.imread("noisy_img.jpg",as_gray=True))
img_median =median (p, mode='constant', cval=0.0)
s = io.imsave("img_median.jpg",img_median)
cv2.imshow("img", img)
cv2.imshow("noisy_img", noisy_img)
cv2.imshow("img_median", img_median)
cv2.waitKey (0)
cv2.destroyAllWindows()


