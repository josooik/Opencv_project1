# Opencv 차선인식(사다리꼴 영역지정 후 차선 추출)
import numpy as np
import cv2

trap_bottom_width = 0.8
trap_top_width = 0.1
trap_height = 0.4

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
writer1 = cv2.VideoWriter('process.mp4', codec, fps, (3840,2160))

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

    cont = cv2.hconcat([img_frame, capture_grays, capture_hsv])
    cont1 = cv2.hconcat([img_dst_y, img_dst_w, img_dst_yw])
    cont2 = cv2.hconcat([img_zero, img_frames_poly, img_poly])
    cont3 = cv2.vconcat([cont, cont1, cont2])

    # 비디오를 저장할때는 반드시 COLOR로 바꿔주어야 함
    # Gray나 binary는 1채널이므로 저장이 안됨
    writer.write(img_poly)
    writer1.write(cont3)

    '''
    txt_position = (10, 30)
    fontScale = 0.8
    myStr = f'{video_counter}/{video_length}'
    img_dst = cv2.putText(img_dst, myStr, txt_position,
                          cv2.FONT_HERSHEY_COMPLEX, fontScale, 255, thickness=2)
    '''

    img_frame2 = cv2.pyrDown(cont3)
    img_frame3 = cv2.pyrDown(img_frame2)

    cv2.imshow('Video', img_frame3)
    cv2.imshow('img_poly', img_poly)

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