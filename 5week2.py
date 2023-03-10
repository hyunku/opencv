import numpy as np
import cv2

# blue, green, red = (255, 0, 0), (0, 255, 0), (0, 0, 255)    	# 색상 선언
# image = np.zeros((400, 600, 3), np.uint8)    					# 3채널 컬러 영상 생성
# image[:] = (255, 255, 255)
#
# pt1, pt2 = (50, 50), (250, 150)                   		        # 좌표 선언 – 정수형 튜플
# pt3, pt4 = (400, 150), (500,  50)
# roi = 50, 200, 200, 100
#
# # 직선 그리기
# cv2.line(image, pt1, pt2, red)
# cv2.line(image, pt3, pt4, green, 3, cv2.LINE_AA)    			# 계단 현상이 감소한 선
#
# # 사각형 그리기
# cv2.rectangle(image, pt1, pt2, blue, 3, cv2.LINE_4)             # 4방향 연결선
# cv2.rectangle(image, roi, red, 3, cv2.LINE_8 )                  # 내부 채움
# cv2.rectangle(image, (400, 200, 100, 100), green, cv2.FILLED )  # 내부 채움
#
# cv2.imshow('Line & Rectangle', image)							# 윈도우에 영상 표시
# cv2.waitKey(0)
# cv2.destroyAllWindows()


olive, violet, brown = (128,128,0), (221,160,221), (42,42,165)
pt1, pt2 = (50, 230), (50,310)

image = np.zeros((350,500,3), np.uint8)
image.fill(255)

cv2.putText('Line & Rectangle', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, brown)
cv2.putText('Line', (50,130), cv2.FONT_HERSHEY_DUPLEX, 3, olive)
cv2.putText('Rectangle', pt1, cv2.FONT_HERSHEY_TRIPLEX, 2, violet)

cv2.imshow('putText', image)
cv2.waitKey(0)