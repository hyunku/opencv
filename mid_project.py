import matplotlib.pyplot as plt
import cv2
import numpy as np

isDragging = False  # 마우스 드래그 상태 저장
x0, y0, w, h = -1, -1, -1, -1  # 영역 선택 좌표 저장
blue, red = (255, 0, 0), (0, 0, 255)  # 색상 값



def onMouse(event, x, y, flags, param):  # 마우스 이벤트 핸들 함수
    global isDragging, x0, y0, img  # 전역변수 참조
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 마우스 버튼 다운(누름) -> 드래그 시작
        isDragging = True
        x0 = x
        y0 = y
    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 움직임
        if isDragging:  # 드래그 진행 중
            img_draw = img.copy()  # 사각형 그림 표현을 위한 이미지 복제
            cv2.rectangle(img_draw, (x0, y0), (x, y), blue, 2)  # 드래그 진행 영역 표시
            cv2.imshow('img', img_draw)  # 사각형 표시된 그림 화면 출력
    elif event == cv2.EVENT_LBUTTONUP:  # 왼쪽 마우스 버튼 업(떼기)
        if isDragging:  # 드래그 중지
            isDragging = False
            w = x - x0  # 드래그 영역 폭 계산
            h = y - y0  # 드래그 영역 높이 계산
            if w > 0 and h > 0:  # 폭과 높이가 음수이면 드래그 방향이 옳음
                img_draw = img.copy()  # 선택 영역에 사각형 그림을 표시할 이미지 복제
                # 선택 영역에 빨간 사각형 표시
                cv2.rectangle(img_draw, (x0, y0), (x, y), red, 2)
                text = "x_pos:{}, y_pos:{}, d_h:{}, d_w:{}".format(x, y, h, w)
                cv2.putText(img_draw, text, (x0 + 10, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=blue, thickness=1)
                cv2.imshow('img', img_draw)  # 빨간 사각형 그려진 이미지 화면 출력
                roi = img[y0:y0 + h, x0:x0 + w]  # 원본 이미지에서 선택 영영만 ROI로 지정
                cv2.imshow('', roi)  # ROI 지정 영역을 새창으로 표시
                cv2.moveWindow('cropped', 0, 0)  # 새창을 화면 좌측 상단에 이동
                # cv2.imwrite('images/cropped.jpg', roi)  # ROI 영역만 파일로 저장

                plt.imshow(roi[:, :, ::-1])  # 잘라낸 영역 보여줌
                plt.xticks([])
                plt.yticks([])
                plt.show()
            else:
                cv2.imshow('img', img)  # 드래그 방향이 잘못된 경우 사각형 그림이 없는 원본 이미지 출력


def onChange(value): # 트랙바 이벤트 - 밝기 조절
    t = (value*5, value*5, value*5) # 레벨 당 5배수단위만큼 조절
    arr = np.full(img2.shape,t,dtype=np.uint8) # 특정 화소만큼의 동일한 사이즈의 그림 생성
    dst = cv2.subtract(img2,arr) # 원본 이미지에서 특정 화소만큼의 차이 계산
    cv2.imshow('brightness',dst) # 밝기조절 완료


img = cv2.imread('images/panda.JPG') #################### 경로 확인 필수 #########################
img2 = img.copy()
cv2.imshow('img', img)
cv2.setMouseCallback('img', onMouse)  # 마우스 이벤트 등록
cv2.imshow('brightness', img2)
cv2.createTrackbar('level','brightness',0,20,onChange) # 트랙바 이벤트 등록
cv2.waitKey()
cv2.destroyAllWindows()

while True:
    img_rot = img.copy()
    rows, cols = img_rot.shape[0:2]
    b, g, r = cv2.split(img_rot)

    # 각도계산
    d45 = 45.0 * np.pi / 180  # 45도
    d90 = 90.0 * np.pi / 180  # 90도

    # 회전변환을 위한 행렬 생성
    m45 = np.float32([[np.cos(d45), -1 * np.sin(d45), rows // 2],
                  [np.sin(d45), np.cos(d45), -1 * cols // 4]])
    m90 = np.float32([[np.cos(d90), -1 * np.sin(d90), rows],
                  [np.sin(d90), np.cos(d90), 0]])

    # 키보드 이벤트 수행할 이미지
    cv2.imshow("rotate image",img_rot)

    # 키보드 이벤트 로직
    key = cv2.waitKey(0) & 0xFF # 키보드 입력까지 대기
    if key == ord('q'): #
        r45 = cv2.warpAffine(img_rot, m45, (cols, rows)) # 변환각만큼 회전
        cv2.imshow("45 rotate", r45)
    elif key == ord('w'):
        r90 = cv2.warpAffine(img_rot, m90, (rows, cols)) # 변환각만큼 회전
        cv2.imshow("90 rotate", r90)
    elif key == ord('e'):
        x_versus = cv2.flip(img_rot, 0) # x축 반전
        cv2.imshow("x flip", x_versus)
    elif key == ord('r'):
        y_versus = cv2.flip(img_rot, 1) # y축 반전
        cv2.imshow("y flip", y_versus)
    elif key == ord('a'): # blue 채널 화소 증가
        cv2.add(b,100,b)
        b_chan_plus = cv2.merge([b,g,r])
        cv2.imshow("blue chan",b_chan_plus)
    elif key == ord('s'): # green 채널 화소 증가
        cv2.add(g,100,g)
        g_chan_plus = cv2.merge([b,g,r])
        cv2.imshow("green chan",g_chan_plus)
    elif key == ord('d'): # red 채널 화소 증가
        cv2.add(r,100,r)
        r_chan_plus = cv2.merge([b,g,r])
        cv2.imshow("red chan",r_chan_plus)
    elif key == ord('y'):
        logo = cv2.imread("images/images/logo.jpg", cv2.IMREAD_COLOR)

        masks = cv2.threshold(logo, 220, 255, cv2.THRESH_BINARY)[1]
        masks = cv2.split(masks)

        fg_pmask = cv2.bitwise_or(masks[0], masks[1])
        fg_pmask = cv2.bitwise_or(masks[2], fg_pmask)
        bg_pmask = cv2.bitwise_not(fg_pmask)

        (H, W), (h, w) = img_rot.shape[:2], logo.shape[:2]
        x, y = (W - w) // 2, (H - h) // 2
        roi = img_rot[y:y + h, x:x + w]

        foreground = cv2.bitwise_and(logo, logo, mask=fg_pmask)
        background = cv2.bitwise_and(roi, roi, mask=bg_pmask)

        result = cv2.add(background, foreground)
        img_rot[y:y+h,x:x+w] = result
        cv2.imshow("masked", img_rot)
    elif key == 27: # esc 키 눌러서 종료
        break
cv2.destroyWindow('rotate image')

