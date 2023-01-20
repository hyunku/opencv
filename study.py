import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("images/images/bit_test.jpg", cv2.IMREAD_COLOR)
logo = cv2.imread("images/images/logo.jpg", cv2.IMREAD_COLOR)

masks = cv2.threshold(logo, 220, 255, cv2.THRESH_BINARY)[1]
masks = cv2.split(masks)

fg_pmask = cv2.bitwise_or(masks[0], masks[1])
fg_pmask = cv2.bitwise_or(masks[2], fg_pmask)
bg_pmask = cv2.bitwise_not(fg_pmask)

(H,W), (h,w) = image.shape[:2], logo.shape[:2]
x,y = (W-w)//2, (H-h)//2
roi = image[y:y+h,x:x+w]

foreground = cv2.bitwise_and(logo, logo, mask=fg_pmask)
background = cv2.bitwise_and(roi, roi, mask=bg_pmask)

cv2.imshow("background", background)
cv2.imshow("foreground", foreground)
cv2.imshow("fg_pmask", fg_pmask)
cv2.imshow("logo", logo)
cv2.waitKey()
