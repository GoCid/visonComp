import cv2
import imutils

print("opencv version:")
print(cv2.__version__)

img = cv2.imread("phoyo.jpg")

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

rot= imutils.rotate_bound(gray,60)

cv2.imshow("Phoyo", img)
cv2.imshow("Phyo - Gray", rot)

cv2.waitKey(0)
cv2.destroyAllWindows