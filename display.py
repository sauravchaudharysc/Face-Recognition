import cv2

#To simple read image
img = cv2.imread('itachi.jpg')

#Read a gray scale image
gray = cv2.imread('itachi.jpg',cv2.IMREAD_GRAYSCALE)

cv2.imshow('Itachi Uchiha',img)
cv2.imshow('Grey Itachi Uchiha',gray)
#this is the basically the time, zero means infinite time
cv2.waitKey(0)
cv2.destroyAllWindows()