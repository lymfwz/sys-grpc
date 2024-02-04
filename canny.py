import numpy as np
import cv2

img1 = cv2.imread(r"photoorg/p1.jpg", 0)
print(img1.dtype)
#print(img1.shape)
#img1_blur = cv2.GaussianBlur(img1,(1,1),0)
#edges = cv2.Canny(img1,100,150)
#contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#line_img = cv2.drawContours(img1, contours, -1, (0, 0, 255), 1)
#print(edges.shape)
#cv2.imshow('edges',edges)
#cv2.imwrite("D:\\yycodes\\canny\\photocanny\\1101_canny.png", line_img)


#gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
canvas = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
canvas[:] = (0,0,0)
cv2.drawContours(canvas, contours, -1, (255, 255, 255),5 )
#edges = cv2.Canny(gray,100,150)
#cv2.imwrite("D:\\yycodes\\canny\\photocanny\\1101_canny.png", edges)
print(canvas.dtype)
cv2.imwrite("./photocanny/p1_canny.png", canvas)