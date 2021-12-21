import cv2
import timeit
import face_recognition
import numpy as np
import os
import glob
import moviepy.editor as mp

# 모자이크
def mosaic(src, ratio):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    mosaic_img = cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    return mosaic_img

# 영역 모자이크
def mosaic_area(src, x, y, width, height, ratio):
    mosaic_area_img = src.copy()
    mosaic_area_img[y:y + height, x:x + width] = mosaic(mosaic_area_img[y:y + height, x:x + width], ratio)
    return mosaic_area_img

# 영상 검출기
def videoDetector(cam, cascade, img_array, FPSs):
    # 이미지 가져오기
    faces_encodings = []
    faces_names = []
    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, "faces/")
    list_of_files = [f for f in glob.glob(path + "*.jpg")]
    number_files = len(list_of_files)
    names = list_of_files.copy()

    # 얼굴 훈련
    for i in range(number_files):
        globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
        globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
        faces_encodings.append(globals()['image_encoding_{}'.format(i)])
        # 이름 배열 생성
        names[i] = names[i].replace(cur_direc, "")
        faces_names.append(names[i])

    # 얼굴 인식
    face_encodings = []
    process_this_frame = True
    while True:
        # 연산 시작
        start_t = timeit.default_timer()

        # 캡처 이미지 불러오기
        ret, img = cam.read()
        # 영상 압축
        try:
            img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
        except:
            break
        # 그레이 스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cascade 얼굴 탐지 알고리즘
        results = cascade.detectMultiScale(gray,  # 입력 이미지
                                           scaleFactor=1.5,  # 이미지 피라미드 스케일 factor
                                           minNeighbors=1,  # 인접 객체 최소 거리 픽셀
                                           minSize=(30, 30)  # 탐지 객체 최소 크기
                                           )

        # 얼굴 인식 과정
        if process_this_frame:
            face_locations = face_recognition.face_locations(img)
            face_encodings = face_recognition.face_encodings(img, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(faces_encodings, face_encoding)
            name = 'Unknown'

        face_distances = face_recognition.face_distance(faces_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = faces_names[best_match_index]

        face_names.append(name)
        process_this_frame = not process_this_frame

        for box in results:
            x, y, w, h = box
            if name == 'Unknown':
                img = mosaic_area(img, x, y, w, h, ratio=0.04)

        # 연산 종료
        terminate_t = timeit.default_timer()
        # 프레임 계산
        FPSs.append(int(1. / (terminate_t - start_t)))

        # 영상 이미지 저장
        img_array.append(img)
        # 영상 출력
        cv2.imshow('facenet', img)

        if cv2.waitKey(1) > 0:
            break

# 결과 폴더 생성
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        pass

if __name__ == '__main__':
    # 가중치 파일 경로
    cascade_filename = "model/haarcascade_frontalface_alt.xml"

    # 모델 불러오기
    cascade = cv2.CascadeClassifier(cascade_filename)

    # 영상 파일
    cam = cv2.VideoCapture("video/sample.mp4")
    # 소리 추출
    videoclip = mp.VideoFileClip("video/sample.mp4")
    videoclip.audio.write_audiofile("video/audio.mp3")

    img_array = []
    FPSs= []

    # 영상 탐지기
    videoDetector(cam, cascade, img_array, FPSs)

    # 결과 폴더 생성
    createFolder("result")

    # 영상 생성 과정
    for img in img_array:
        height, width, layers = img.shape
        size = (width, height)

    sum = 0
    count = 0
    for fps in FPSs:
        sum += fps
        count += 1
    avg = sum/count

    out = cv2.VideoWriter("result/result.mp4", cv2.VideoWriter_fourcc(*'DIVX'), avg/2, size)
    for i in range(len(img_array)):
        out.write(img_array[i])

    # 영상 해제
    out.release()
    cam.release()
    cv2.destroyAllWindows()

    # 오디오 합치기
    videoclip2 = mp.VideoFileClip("result/result.mp4")
    audioclip2 = mp.AudioFileClip("video/audio.mp3")

    videoclip2.audio = audioclip2
    videoclip2.write_videofile("result/resultVideo.mp4")

    videoclip.close()
    videoclip2.close()
    audioclip2.close()

    os.remove(r"video/audio.mp3")
    os.remove(r"result/result.mp4")