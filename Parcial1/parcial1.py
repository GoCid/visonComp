from multiprocessing.connection import wait
import cv2
import imutils
import numpy as np
from cv2 import getStructuringElement


#CARGAR IMAGEN ORIGINAL
img = cv2.imread('examen_b.tif')
#cv2.imshow('Img Original', img)


img= cv2.bilateralFilter(img,5,6,6)
img_gaus=cv2.GaussianBlur(img,(3,3),2)
img=cv2.addWeighted(img,7.5, img_gaus,-6.5,0)
#cv2.imshow('nita',img)

#SOBEL
#gradX = cv2.Sobel(img,-1,1,0,ksize=3)
#gradY = cv2.Sobel(img,-1,0,1,ksize=3)
#img_sobel = cv2.addWeighted(gradX, 0.5, gradY, 0.5,0)
#cv2.imshow("Sobel", img_sobel)
#cv2.imwrite("Sobel.tif", img_sobel)


#ROTAR IMAGEN
img = imutils.rotate(img,-15)
#cv2.imshow('Rotada',img)
#cv2.imwrite('Rotada.tif', img)

#RECORTAR
img = img[82:428, 245:595]
#cv2.imshow("Recortada", img)
#cv2.imwrite('Recortada.tif', img)

#DILATACION intentando hacer que el rostro se vea mejor
kernel = getStructuringElement(cv2.MORPH_RECT,(3,3))

img = cv2.erode(img, kernel, iterations=1)
img = cv2.dilate(img,kernel,iterations=2)

img = cv2.dilate(img, kernel, iterations=1)
img = cv2.erode(img, kernel, iterations=2)
#cv2.imshow("Dilatacion", img)
#cv2.imwrite("Dilatacion.tif", img)

#AGREGANDO NITIDEZ

img=cv2.bilateralFilter(img,3,6,6)
img_gaus= cv2.GaussianBlur(img,(3,3),2)

#nit=cv2.addWeighted(img,1.5, img_gaus,-0.5,0)
#nit=cv2.addWeighted(img,3.5, img_gaus,-2.5,0)
nit=cv2.addWeighted(img,7.5, img_gaus,-6.5,0)
nit=cv2.bilateralFilter(nit,5,6,6)

#kernel = np.array([[-1, -1, -1],
#                   [-1, 9,-1],
#                  [-1, -1, -1]])
#img=cv2.filter2D(img,ddepth=-1,kernel=kernel)
#gradX = cv2.Sobel(img,-1,1,0,ksize=3)
#gradY = cv2.Sobel(img,-1,0,1,ksize=3)
#img_sobel = cv2.addWeighted(gradX, 0.5, gradY, 0.5,0)
#img=cv2.addWeighted(img_sobel,0.9,img,0.9,0)
#kernel = getStructuringElement(cv2.MORPH_RECT,(3,3))
#img=cv2.dilate(img,kernel,iterations=1)
#img=cv2.erode(img,kernel,iterations=1)

#cv2.imshow("Sobel", img_sobel)
cv2.imwrite('NitidaFINAL.tif',nit)
cv2.imshow('nit',nit)
#END
cv2.waitKey(0)
cv2.destroyAllWindows d