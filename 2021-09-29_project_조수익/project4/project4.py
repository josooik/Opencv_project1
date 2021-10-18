# Opencv 차선인식(project1 + project2 + project3 + project4)
import numpy as np
import cv2

trap_bottom_width = 0.8
trap_top_width = 0.1
trap_height = 0.4

rho = 10
theta = 1 * np.pi / 180
threshold = 50
min_line_length = 10
max_line_gap = 30

def draw_lines(img, lines, color=[255, 0, 0], thickness=12):
    # 예외처리
    if lines is None:
        return
    if len(lines) == 0:
        return

    draw_right = True
    draw_left = True

    # 모든 선의 기울기 찾기
    # 기울기의 절대값이 임계값 보다 커야 추출됨
    slope_threshold = 0.5
    slopes = []
    new_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]
        # 기울기 계산
        if x2 - x1 == 0.:  # 기울기의 분모가 0일때 예외처리
            slope = 999.  # practically infinite slope
        else:
            slope = (y2 - y1) / (x2 - x1)
        # 조건을 만족하는 line을 new_lines에 추가
        if abs(slope) > slope_threshold:
            slopes.append(slope)
            new_lines.append(line)
    lines = new_lines

    # 라인을 오른쪽과 왼쪽으로 구분하기
    # 기울기 및 점의 위치가 영상의 가운데를 기준으로 왼쪽 또는 오른쪽에 위치
    right_lines = []
    left_lines = []

    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        img_x_center = img.shape[1] / 2  # 영상의 가운데 x 좌표
        # 기울기 방향이 바뀐이유는 y축의 방향이 아래로 내려오기 때문
        if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
            right_lines.append(line)
        elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
            left_lines.append(line)

    # np.polyfit()함수를 사용하기 위해 점들을 추출
    # Right lane lines
    right_lines_x = []
    right_lines_y = []

    for line in right_lines:
        x1, y1, x2, y2 = line[0]
        right_lines_x.append(x1)
        right_lines_x.append(x2)
        right_lines_y.append(y1)
        right_lines_y.append(y2)

    if len(right_lines_x) > 0:
        right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)  # y = m*x + b
    else:
        right_m, right_b = 1, 1
        draw_right = False

    # Left lane lines
    left_lines_x = []
    left_lines_y = []

    for line in left_lines:
        x1, y1, x2, y2 = line[0]
        left_lines_x.append(x1)
        left_lines_x.append(x2)
        left_lines_y.append(y1)
        left_lines_y.append(y2)

    if len(left_lines_x) > 0:
        left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b
    else:
        left_m, left_b = 1, 1
        draw_left = False

    # 왼쪽과 오른쪽의 2개의 점을 찾기, y는 알고 있으므로 x만 찾으면됨
    # y = m*x + b --> x = (y - b)/m
    y1 = img.shape[0]
    y2 = img.shape[0] * (1 - trap_height)
    right_x1 = (y1 - right_b) / right_m
    right_x2 = (y2 - right_b) / right_m
    left_x1 = (y1 - left_b) / left_m
    left_x2 = (y2 - left_b) / left_m

    # 모든 점은 정수형이어야 함(정수형으로 바꾸기)
    y1 = int(y1)
    y2 = int(y2)
    right_x1 = int(right_x1)
    right_x2 = int(right_x2)
    left_x1 = int(left_x1)
    left_x2 = int(left_x2)

    # 차선그리기
    if draw_right:
        cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
    if draw_left:
        cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)

capture = cv2.VideoCapture('mov/challenge.mp4')
play_mode = 1 # 0: play once 1:play continuously

if capture.isOpened() == False:
  print("카메라를 열 수 없습니다.")
  exit(1)

video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
codec = cv2.VideoWriter_fourcc('m','p','4','v') # .mp4
#codec = cv2.VideoWriter_fourcc('M','J','P','G') # .avi

fps = 30.0
# 동영상 파일을 저장하려면 VideoWrite객체를 생성
# VideoWriter객체를 초기화 하기 위해 저장할 동영상 파일 이름,
# 코덱, 프레임레이트, 이미지 크기를 지정해야함
writer = cv2.VideoWriter('output.mp4', codec, fps, (width,height))
writer1 = cv2.VideoWriter('process.mp4', codec, fps, (6400, 2160), isColor=True)

#VideoWriter객체를 성공적으로 초기화 했는지 체크
if writer.isOpened() == False:
   print('동영상 저장파일객체 생성하는데 실패하였습니다.')
   exit(1)

if writer1.isOpened() == False:
   print('동영상 저장파일객체 생성하는데 실패하였습니다.')
   exit(1)

#Esc키를 눌러 동영상을 중단하면 종료직전까지 동영상이 저장됨
video_counter = 0

if capture.isOpened() == False:
    print("동영상을 열수없습니다.")
    exit(1)

while True:
    ret, img_frame = capture.read()

    img_frames = img_frame.copy()
    img_frames1 = img_frame.copy()
    capture_gray = cv2.cvtColor(img_frames, cv2.COLOR_BGR2GRAY)
    capture_grays = cv2.cvtColor(capture_gray, cv2.COLOR_GRAY2BGR)

    if play_mode == 0:
        if ret == False:  # 동영상 끝까지 재생
            print("동영상 읽기 완료")
            break
    elif play_mode == 1:  # 동영상이 끝나면 재생되는 프레임의 위치를 0으로 다시 지정
        if capture.get(cv2.CAP_PROP_POS_FRAMES) == \
                capture.get(cv2.CAP_PROP_FRAME_COUNT):
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    capture_hsv = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)
    img_h, img_S, img_v = cv2.split(capture_hsv)

    # HSV로 노랑색 정보를 좀 더 구체적으로 표시
    lower_yellow = (20, 125, 130)  # 자료형은 튜플형태로(H, S, V)
    upper_yellow = (40, 255, 255)  # 자료형은 튜플형태로(H, S, V)

    img_mask_y = cv2.inRange(capture_hsv, lower_yellow, upper_yellow)  # 노랑색 정보 추출(특정 범위 안에 있는 행렬 원소 검출)
    img_mask_ys = cv2.merge((img_mask_y, img_mask_y, img_mask_y))
    img_dst_y = cv2.bitwise_and(img_frame, img_frame, mask=img_mask_y)  # AND 비트연산

    # HSV로 하얀색 정보를 좀 더 구체적으로 표시
    img_dst_w = np.copy(img_frame)
    bgr_threshold = [200, 200, 200]

    # BGR 제한 값보다 작으면 검은색으로
    thresholds = (img_frame[:, :, 0] < bgr_threshold[0]) \
                 | (img_frame[:, :, 1] < bgr_threshold[1]) \
                 | (img_frame[:, :, 2] < bgr_threshold[2])
    img_dst_w[thresholds] = [0, 0, 0]

    img_dst_yw = cv2.addWeighted(img_dst_y, 1.0, img_dst_w, 1.0, 0)

    img_zero = np.zeros_like(img_frames)
    height, width = img_zero.shape[:2]

    pts = np.array([[
        ((width * (1 - trap_bottom_width)) // 2, height),
        ((width * (1 - trap_top_width)) // 2, (1 - trap_height) * height),
        (width - (width * (1 - trap_top_width)) // 2, (1 - trap_height) * height),
        (width - (width * (1 - trap_bottom_width)) // 2, height)]],
        dtype=np.int32)

    cv2.fillPoly(img_zero, pts, (255, 255, 255), cv2.LINE_AA)

    img_frames_poly = cv2.bitwise_and(img_frames, img_zero)
    img_poly = cv2.bitwise_and(img_dst_yw, img_zero)

    # 블러(흐림)을 사용해서 노이즈 제거
    img_gray1 = cv2.cvtColor(img_dst_yw, cv2.COLOR_BGR2GRAY)
    img_grays1 = cv2.cvtColor(img_gray1, cv2.COLOR_GRAY2BGR)

    img_gauss = cv2.GaussianBlur(img_gray1, (1, 1), 0)
    img_gausss = cv2.merge((img_gauss, img_gauss, img_gauss))

    # 임계값(Threshold)를 사용하여 이진화
    _, frame_binary = cv2.threshold(img_gauss, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    frame_binarys = cv2.merge((frame_binary, frame_binary, frame_binary))

    # 외각선(엣지) 구하기 : Canny 엣지를 사용
    frame_canny = cv2.Canny(frame_binary, 50, 150)
    frame_cannys = cv2.merge((frame_canny, frame_canny, frame_canny))

    img_zero1 = np.zeros_like(frame_canny)
    height, width = img_zero.shape[:2]

    pts = np.array([[
        ((width * (1 - trap_bottom_width)) // 2, height),
        ((width * (1 - trap_top_width)) // 2, (1 - trap_height) * height),
        (width - (width * (1 - trap_top_width)) // 2, (1 - trap_height) * height),
        (width - (width * (1 - trap_bottom_width)) // 2, height)]],
        dtype=np.int32)

    cv2.fillPoly(img_zero1, pts, 255)

    img_roi = cv2.bitwise_and(frame_canny, img_zero1)  # 관심영역
    img_rois = cv2.merge((img_roi, img_roi, img_roi))

    lines = cv2.HoughLinesP(img_roi, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    draw_lines(img_frames1, lines, (0, 255, 0), 12)

    cont = cv2.hconcat([img_frame, capture_grays, capture_hsv, img_dst_y, img_dst_w])
    cont1 = cv2.hconcat([img_dst_yw, img_zero, img_frames_poly, img_poly, img_grays1])
    cont2 = cv2.hconcat([img_gausss, frame_binarys, frame_cannys, img_rois, img_frames1])
    cont3 = cv2.vconcat([cont, cont1, cont2])

    # 비디오를 저장할때는 반드시 COLOR로 바꿔주어야 함
    # Gray나 binary는 1채널이므로 저장이 안됨

    '''
    txt_position = (10, 30)
    fontScale = 0.8
    myStr = f'{video_counter}/{video_length}'
    img_dst = cv2.putText(img_dst, myStr, txt_position,
                          cv2.FONT_HERSHEY_COMPLEX, fontScale, 255, thickness=2)
    '''

    writer.write(img_frames1)
    writer1.write(cont3)

    img_frame3 = cv2.pyrDown(cont3)
    img_frame4 = cv2.pyrDown(img_frame3)
    cv2.imshow('img_frame4', img_frame4)
    cv2.imshow('Video', img_frames1)

    if video_counter == video_length:
        video_counter = 0
    else:
        video_counter += 1

    # 동영상이 끝나면 재생되는 프레임의 위치를 0으로 다시 지정
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    key = cv2.waitKey(33)
    if key == 27:  # ESC 키
        break

capture.release()
writer.release()
writer1.release()
cv2.destroyAllWindows()