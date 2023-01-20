import matplotlib.pyplot as plt
import numpy as np
import cv2

x = np.arange(10)
y1 = np.arange(10)
y2 = np.arange(10) ** 2
y3 = np.random.choice(50, size=10)

plt.figure(figsize=(5,3))
plt.plot(x,y1,'b--',linewidth=2)
plt.plot(x,y2,'go-',linewidth=3)
plt.plot(x,y3,'c+:',linewidth=5)

plt.title("Line examples")
plt.axis([0,10,0,80])
plt.tight_layout()
# plt.savefig(fname='sample.png', dpi=300)
plt.show()

image = cv2.imread("images/images/flip_test.jpg", cv2.IMREAD_COLOR)
if image is None: raise Exception("영상 파일 읽기 오류 발생") # 예외 처리

x_axis = cv2.flip(image, 0)                 # x축 기준 상하 뒤집기
y_axis = cv2.flip(image, 1)                 # y축 기준 좌우 뒤집기
xy_axis = cv2.flip(image, -1)
rep_image = cv2.repeat(image, 1, 2)       # 반복 복사
trans_image = cv2.transpose(image)          # 행렬 전치

## 각 행렬을 영상으로 표시
titles = ['image', 'x_axis', 'y_axis','xy_axis','rep_image','trans_image']
for title in titles:
    cv2.imshow(title, eval(title))
cv2.waitKey(0)

image = cv2.imread("images/images/color.jpg", cv2.IMREAD_COLOR)

bgr = cv2.split(image)
print("bgr자료형:", type(bgr), type(bgr[0]), type(bgr[0][0][0]))
print("원소 개수:", len(bgr))

cv2.imshow("image", image)
cv2.imshow("blue_chan", bgr[0])
cv2.imshow("green_chan", bgr[1])
cv2.imshow("red_chan", bgr[2])
cv2.waitKey(0)

image1 = np.zeros((300, 300), np.uint8)     		# 300행, 300열 검은색 영상 생성
image2 = image1.copy()

h, w = image1.shape[:2]
cx,cy  = w//2, h//2
cv2.circle(image1, (cx,cy), 100, 255, -1)      		# 중심에 원 그리기
cv2.rectangle(image2, (0,0, cx, h), 255, -1)

image3 = cv2.bitwise_or(image1, image2)     	# 원소 간 논리합
image4 = cv2.bitwise_and(image1, image2)    	# 원소 간 논리곱
image5 = cv2.bitwise_xor(image1, image2)    	# 원소 간 배타적 논리합
image6 = cv2.bitwise_not(image1)            	# 행렬 반전

cv2.imshow("image1", image1);			cv2.imshow("image2", image2)
cv2.imshow("bitwise_or", image3);		cv2.imshow("bitwise_and", image4)
cv2.imshow("bitwise_xor", image5);	cv2.imshow("bitwise_not", image6)
cv2.waitKey(0)