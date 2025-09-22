import numpy as np
import cv2 as cv
import sys
import winsound  # Windows 전용 모듈

# YOLO v3 모델을 초기화하고 클래스 이름을 불러오는 함수
def consturct_yolo_v3():
    # COCO 데이터셋 클래스 이름 파일을 읽음
    f = open('coco_names.txt', 'r')
    class_names = [line.strip() for line in f.readlines()]
    
    # YOLO v3 가중치 파일과 구성 파일을 불러옴
    model = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    
    # YOLO 모델의 출력 레이어 이름을 가져옴
    layer_names = model.getLayerNames()
    out_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]
    
    return model, out_layers, class_names

# 이미지에서 객체를 탐지하는 함수
def yolo_detect(img, yolo_model, out_layers, conf_threshold=0.6, nms_threshold=0.3):
    height, width = img.shape[0], img.shape[1]
    
    # 이미지를 YOLO 입력으로 변환
    blob = cv.dnn.blobFromImage(img, 1.0/255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    yolo_model.setInput(blob)
    
    # YOLO 모델을 사용하여 추론 실행
    output_layers = yolo_model.forward(out_layers)
    
    boxes = []
    confidences = []
    class_ids = []
    
    # 추론 결과를 분석하여 신뢰도가 높은 객체 추출
    for output in output_layers:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # 비최대 억제를 적용하여 중복 박스 제거
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    objects = [[boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3], confidences[i], class_ids[i]] for i in indices.flatten()]
    return objects

# YOLO v3 모델과 클래스 이름을 초기화
model, out_layers, class_names = consturct_yolo_v3()
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

# 이미지를 불러옴
img = cv.imread('knife5.jpg')
if img is None:
    sys.exit('파일이 없습니다.')

# 특정 클래스 ID (칼 ID = 43)
knife_class_id = 43
knife_detected = False

# 이미지에서 객체 탐지 수행
res = yolo_detect(img, model, out_layers)

# 탐지된 객체를 이미지에 그리기
for i in range(len(res)):
    x1, y1, x2, y2, confidence, class_id = res[i]
    if class_id == knife_class_id:
        knife_detected = True
    text = f"{class_names[class_id]} {confidence:.3f}"
    cv.rectangle(img, (x1, y1), (x2, y2), colors[class_id], 2)
    cv.putText(img, text, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)

# 칼이 탐지되었을 경우 경고음과 경고 메시지 표시
if knife_detected:
    winsound.Beep(1000, 500)  # 1000 Hz, 500 ms 경고음
    # 경고 메시지 이미지 생성
    warning_img = np.zeros((200, 400, 3), dtype=np.uint8)
    cv.putText(warning_img, "WARNING: Knife detected!", (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv.imshow("Warning", warning_img)

# 결과 이미지와 경고 메시지 창 표시
cv.imshow("Object detection by YOLO v.3", img)

cv.waitKey(0)
cv.destroyAllWindows()
