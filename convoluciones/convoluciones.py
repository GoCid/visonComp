import cv2
import numpy as np

img=cv2.imread('original.jpg')

# BILATERAL FILTER
bilateral = cv2.bilateralFilter(img, 5, 100, 100)
cv2.imshow('bilateral', bilateral)

# BLUR
#ksize=(40,40)
#blur=cv2.blur(img,ksize)
#cv2.imshow('blur',blur)

# DILATE AND ERODE
#kernel = np.ones((5, 5), np.uint8)
#img_dilation = cv2.dilate(img, kernel, iterations=2)
#img_erosion = cv2.erode(img, kernel, iterations=3)
#cv2.imshow('Erosion', img_erosion)
#cv2.imshow('Dilation', img_dilation)

# GAUSSIAN BLUR
#gaus= cv2.GaussianBlur(img,(5,5),cv2.BORDER_ISOLATED)
#cv2.imshow("Gaussian Smoothing", gaus)

# MEDIAN FILTER
#median = cv2.medianBlur(img,13)
#cv2.imshow('median', median)

# SPATIALGRADIENT
#sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
#sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
#cv2.imshow('spatialgradient',sobelx)
#cv2.imshow('grad', sobely)
cv2.waitKey(0)
cv2.destroyAllWindows