import numpy as np
import cv2

img1 = cv2.imread("./photocanny/p1_canny.png")
print(img1.dtype)
img2 = cv2.imread("./photo/test.png")
print(img2.dtype)

rows, cols, chn = img1.shape
#r1 = cv2.split(img1)[0]
#b1 = cv2.split(img1)[0]
g1 = cv2.split(img1)[0]
# G, R通道设置为0,
r1 = np.zeros((rows, cols), dtype=img1.dtype)
b1 = np.zeros((rows, cols), dtype=img1.dtype)
#g1 = np.zeros((rows, cols), dtype=img1.dtype)
m1 = cv2.merge([r1, g1, b1])

img = cv2.add(m1,img2)
cv2.imwrite("./photofinal/p1_final.png", img)