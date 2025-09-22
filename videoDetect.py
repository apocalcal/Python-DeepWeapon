import numpy as np
import cv2 as cv
import sys
import winsound  # Windows 전용 모듈

# YOLO v3 모델을 구성하는 함수
def consturct_yolo_v3():
    # 클래스 이름을 파일에서 읽어옴
    f = open('coco_names.txt', 'r')
    class_names = [line.strip() for line in f.readlines()]
    
    # YOLO 모델 가중치와 설정 파일 로드
    model = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    layer_names = model.getLayerNames()
    out_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]
    
    return model, out_layers, class_names

# 이미지를 받아 YOLO로 객체를 탐지하는 함수
def yolo_detect(img, yolo_model, out_layers):
    height, width = img.shape[0], img.shape[1]
    # 이미지를 블롭 형태로 변환
    blob = cv.dnn.blobFromImage(img, 1.0/255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    
    yolo_model.setInput(blob)
    output_layers = yolo_model.forward(out_layers)
    
    boxes = []
    confidences = []
    class_ids = []
    
    # 출력 레이어를 순회하며 탐지된 객체 정보 추출
    for output in output_layers:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # 신뢰도가 0.5 이상인 경우에만 처리
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # 비최대 억제 (Non-Maximum Suppression)을 사용하여 박스를 필터링
    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    objects = [[boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3], confidences[i], class_ids[i]] for i in indices.flatten()]
    return objects

# YOLO v3 모델과 필요한 레이어, 클래스 이름 로드
model, out_layers, class_names = consturct_yolo_v3()
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

# 웹캠을 통해 영상 캡처
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened():
    sys.exit('카메라 연결 실패')

# 칼의 클래스 ID (coco.names 파일에서 "knife"의 ID)q
knife_class_id = 43

# 비디오 스트림에서 프레임을 읽고 객체 탐지 수행
while True:
    ret, frame = cap.read()
    if not ret:
        sys.exit('프레임 획득에 실패하여 루프를 나갑니다.')
    
    res = yolo_detect(frame, model, out_layers)
    
    knife_detected = False

    # 탐지된 객체들을 프레임에 표시
    for i in range(len(res)):
        x1, y1, x2, y2, confidence, class_id = res[i]
        if class_id == knife_class_id:
            knife_detected = True
        text = f"{class_names[class_id]} {confidence:.3f}"
        cv.rectangle(frame, (x1, y1), (x2, y2), colors[class_id], 2)
        cv.putText(frame, text, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)
    
    5# 칼이 탐지되었을 경우 경고음과 경고 메시지 표시
    if knife_detected:
        winsound.Beep(1000, 500)  # 1000 Hz, 500 ms 경고음
        # 경고 메시지 이미지 생성
        warning_img = np.zeros((200, 400, 3), dtype=np.uint8)
        cv.putText(warning_img, "WARNING: Knife is detected!", (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv.imshow("Warning", warning_img)

    # 객체 탐지 결과를 화면에 표시
    cv.imshow("Object detection from video by YOLO v.3", frame)
    
    # 'q' 키를 누르면 루프 종료
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# 자원 해제
cap.release()
cv.destroyAllWindows()

