import numpy as np
import cv2

image1 = cv2.imread("images/dog.PNG", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("images/cat.PNG", cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread("images/panda.JPG", cv2.IMREAD_GRAYSCALE)

data_blur = [1/9, 1/9, 1/9,
             1/9, 1/9, 1/9,
             1/9, 1/9, 1/9]

data_sharp = [0, -1, 0,
              -1, 5, -1,
              0, -1, 0]

my_blur_data = [1/16, 1/16, 1/16, 1/16,
                1/16, 1/16, 1/16, 1/16,
                1/16, 1/16, 1/16, 1/16,
                1/16, 1/16, 1/16, 1/16]

my_sharp_data = [0, -1, -1, 0,
                 -1, 2, 2, -1,
                 -1, 2, 2, -1,
                 0, -1, -1, 0]

mask_blur = np.array(data_blur, np.float32).reshape(3, 3)
mask_sharp = np.array(data_sharp, np.float32).reshape(3, 3)

mask_blur_my = np.array(my_blur_data, np.float32).reshape(4, 4)
mask_sharp_my = np.array(my_sharp_data, np.float32).reshape(4, 4)

blur1 = cv2.filter2D(image1, -1, mask_blur)
blur2 = cv2.filter2D(image2, -1, mask_blur)
blur3 = cv2.filter2D(image3, -1, mask_blur)
sharp1 = cv2.filter2D(image1, -1, mask_sharp)
sharp2 = cv2.filter2D(image2, -1, mask_sharp)
sharp3 = cv2.filter2D(image3, -1, mask_sharp)

blur1_my = cv2.filter2D(image1, -1, mask_blur_my)
blur2_my = cv2.filter2D(image2, -1, mask_blur_my)
blur3_my = cv2.filter2D(image3, -1, mask_blur_my)
sharp1_my = cv2.filter2D(image1, -1, mask_sharp_my)
sharp2_my = cv2.filter2D(image2, -1, mask_sharp_my)
sharp3_my = cv2.filter2D(image3, -1, mask_sharp_my)


cv2.imshow("image1", image1)
cv2.imshow("image2", image2)
cv2.imshow("image3", image3)
cv2.imshow("blur1", blur1)
cv2.imshow("blur2", blur2)
cv2.imshow("blur3", blur3)
cv2.imshow("sharp1", sharp1)
cv2.imshow("sharp2", sharp2)
cv2.imshow("sharp3", sharp3)

cv2.imshow("blur1_my", blur1_my)
cv2.imshow("blur2_my", blur2_my)
cv2.imshow("blur3_my", blur3_my)
cv2.imshow("sharp1_my", sharp1_my)
cv2.imshow("sharp2_my", sharp2_my)
cv2.imshow("sharp3_my", sharp3_my)



cv2.waitKey(0)
