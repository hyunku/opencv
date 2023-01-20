import matplotlib.pyplot as plt
import cv2
import numpy as np

data_blur = [1/9, 1/9, 1/9,
             1/9, 1/9, 1/9,
             1/9, 1/9, 1/9]

data_sharp = [0, -1, 0,
              -1, 5, -1,
              0, -1, 0]


def calc_spectrum(complex):
    if complex.ndim == 2:
        dst = abs(complex)
    else:
        dst = cv2.magnitude(complex[:,:,0], complex[:,:,1])
    dst = cv2.log(dst + 1)
    cv2.normalize(dst, dst, 0, 255, cv2.NORM_MINMAX)
    return cv2.convertScaleAbs(dst)

def fftshift(img):
    dst = np.zeros(img.shape, img.dtype)
    h,w = dst.shape[:2]
    cy, cx = h // 2, w // 2
    dst[h-cy:, w-cx:] = np.copy(img[0:cy, 0:cx])
    dst[0:cy, 0:cx] = np.copy(img[h-cy:, w-cx:])
    dst[0:cy, w - cx:] = np.copy(img[h-cy:, 0:cx])
    dst[h - cy:, 0:cx] = np.copy(img[0:cy, w-cx:])
    return dst


def scaling(img, size):
    dst = np.zeros(size[::-1], img.dtype)
    ratioY, ratioX = np.divide(size[::-1], img.shape[:2])
    y = np.arange(0, img.shape[0], 1)
    x = np.arange(0, img.shape[1], 1)
    y, x = np.meshgrid(y, x)
    i, j = np.int32(y * ratioY), np.int32(x * ratioX)
    dst[i, j] = img[y, x]
    return dst

def scaling_nearest(img, size):
    dst = np.zeros(size[::-1], img.dtype)
    ratioY, ratioX = np.divide(size[::-1], img.shape[:2])
    i = np.arange(0, size[1], 1)
    j = np.arange(0, size[0], 1)
    i, j = np.meshgrid(i, j)
    y, x = np.int32(i / ratioY), np.int32(j / ratioX)
    dst[i, j] = img[y, x]
    return dst


def onChange(x): # 트랙바 이벤트 - 블렌딩
    alpha = x / 100
    blending = cv2.addWeighted(img, 1 - alpha, dst3, alpha, 0)
    cv2.imshow('blending', blending)


img = cv2.imread('images/panda.JPG') #################### 경로 확인 필수 #########################
img1 = cv2.imread('images/cat.PNG')
img2 = cv2.imread('images/dog.PNG')
img3 = cv2.imread('images/dog.PNG', cv2.IMREAD_GRAYSCALE)

print(f'image shape: {img.shape}') # 494, 557 (h, w)
print(f'image shape: {img2.shape}') # 447, 500 (h, w)

# 보간법(interpolation)
dst1 = scaling(img3, (600, 600))  # 보간없이 확대
dst2 = scaling_nearest(img3, (600, 600))  # NN방법으로 확대
cv2.imshow('non_interpolation', dst1)
cv2.imshow('NN_interpolation', dst2)

cv2.waitKey(0) # 아무 키 누르면
cv2.destroyAllWindows()

dst3 = cv2.resize(img2,(557, 494), interpolation=cv2.INTER_CUBIC) # 3차원 보간법(4by4)
dst4 = cv2.resize(img2,(557, 494), interpolation=cv2.INTER_LANCZOS4) # 4차원 보간법(8by8)
cv2.imshow('original', img2)
cv2.imshow('cubic_interpolation', dst3)
cv2.imshow('lanczos_interpolation', dst4)

cv2.waitKey(0) # 아무 키 누르면
cv2.destroyAllWindows()

# 영상 합성 명암 트랙바
cv2.imshow('blending', img)
cv2.createTrackbar('fade', 'blending', 0, 100, onChange)

cv2.waitKey(0)
cv2.destroyWindow('blending')

# 이퀄라이징
# 히스토그램
cv2.imshow('img', img3)
hist = cv2.calcHist([img], [0], None, [256], [0,256])
plt.plot(hist)
plt.show()

# RGB 히스토그램
cv2.imshow('img', img)

channels = cv2.split(img)
colors = ('b', 'g', 'r')
for (ch, color) in zip (channels, colors):
    hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
    plt.plot(hist, color = color)
plt.show()

# 이퀄라이징
img_yuv = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)

img_eq = img_yuv.copy()
img_eq[:,:,0] = cv2.equalizeHist(img_eq[:,:,0]) # 밝기에 대한 이퀄라이징
img_eq = cv2.cvtColor(img_eq, cv2.COLOR_YUV2BGR)

img_clahe = img_yuv.copy()
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))  #CLAHE 생성
img_clahe[:,:,0] = clahe.apply(img_clahe[:,:,0])            #CLAHE 적용
img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YUV2BGR)

cv2.imshow('Before', img1)
cv2.imshow('equalizeHist', img_eq)
cv2.imshow('CLAHE', img_clahe)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 필터링
mask_blur = np.array(data_blur, np.float32).reshape(3, 3) # 블러링
mask_sharp = np.array(data_sharp, np.float32).reshape(3, 3) # 샤프닝

blur = cv2.filter2D(img3, -1, mask_blur)
sharp = cv2.filter2D(img3, -1, mask_sharp)

cv2.imshow("original", img3)
cv2.imshow("blur", blur)
cv2.imshow("sharp", sharp)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 엣지 검출
# 소벨 마스크, Sobel(이미지, -1(원본크기 유지), x미분차수, y미분차수, 커널크기) 선택파라미터 (미분계수, 연산결과 가산값)
sobelx = cv2.Sobel(img, -1, 1, 0, ksize=3) # x방향 미분 - 수직 마스크
sobely = cv2.Sobel(img, -1, 0, 1, ksize=3)  # y방향 미분 - 수평 마스크
edge = cv2.Laplacian(img, -1) # 라플라시안 필터
edges = cv2.Canny(img,100,200) # 케니 엣지 검출 - Canny(이미지, 최소임계, 최대임계, 결과배열, 소벨커널 크기, 경사하강법 방식)

cv2.imshow("original", img)
cv2.imshow("sobelx", sobelx)
cv2.imshow("sobely", sobely)
cv2.imshow("sobel", sobelx + sobely)
cv2.imshow("laplacian", edge)
cv2.imshow("Canny", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 모폴로지
# 모폴로지용 사각형 커널 (3x3)
k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
erosion = cv2.erode(img, k) # 침식 - 잡음제거
dst = cv2.dilate(img, k) # 팽창 - 객체 내부 잡음 제거
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, k) # 열림 - 침식 후 팽창 -> 노이즈 제거 특화
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k) #  닫힘 - 팽창 후 침식 -> 어두운 노이즈, 연결, 구멍 메우기 특화

cv2.imshow("original", img)
cv2.imshow("erosion", erosion)
cv2.imshow("dilate", dst)
cv2.imshow("open", opening)
cv2.imshow("close", closing)


cv2.waitKey(0)
cv2.destroyAllWindows()

# 푸리에 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT) # dft 로 주파수 영역대 추출
dft_shift = np.fft.fftshift(dft) # shift 진행
out = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])) # log 취함

inverse_shift = np.fft.fftshift(dft_shift) # 역 shift 수행
inverse_dft = cv2.dft(inverse_shift, flags=cv2.DFT_INVERSE) # 역 dft 수행
out2 = cv2.magnitude(inverse_dft[:, :, 0], inverse_dft[:, :, 1])

plt.subplot(131)
plt.imshow(gray, cmap='gray')
plt.title('original')
plt.subplot(132)
plt.imshow(out, cmap='gray')
plt.title('dft')
plt.subplot(133)
plt.imshow(out2, cmap='gray')
plt.title('inverse')

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

x, y, w, h = cv2.selectROI('img', img, False)
if w and h:
    roi = img[y:y + h, x:x + w]
    cv2.imshow('roi', roi)  # ROI 지정 영역을 새창으로 표시
    cv2.moveWindow('roi', 0, 0)  # 새창을 화면 측 상단으로 이동
    # 키보드 이벤트 로직
    key = cv2.waitKey(0) & 0xFF # 키보드 입력까지 대기
    if key == ord('e'):
        x_versus = cv2.flip(roi, 0) # x축 반전
        cv2.imshow("x flip", x_versus)



cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

