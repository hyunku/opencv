import numpy as np
import cv2

# (세로,가로)
image = np.zeros((400,300), np.uint8) # uint8 : 2^8 개 bit 필요함 (gray-scale : 0 ~ 255 변환)
image.fill(255)

cv2.imshow("Window title", image)
cv2.waitKey(0)
cv2.destroyAllWindows()