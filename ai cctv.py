# AI-CCTV

import cv2
import numpy as np
import torch
from pathlib import Path
from yolov5.models.yolo import Model as attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

# 'yolov5/utils/general.py' 파일에서 scale_coords 함수를 가져오기
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    gain = float(img1_shape[0]) / img0_shape[0], float(img1_shape[1]) / img0_shape[1]
    if ratio_pad is not None:  # adjust for aspect ratio and pad
        gain = np.array(gain)
        pad = (img0_shape[0] - img0_shape[1]) * (1 - ratio_pad) / 2  # [x, y] padding
        gain *= img0_shape / (img0_shape - pad * 2)
        coords[:, [0, 2]] -= pad  # x padding
        coords[:, [1, 3]] += pad  # y padding
    coords[:, :4] = coords[:, :4] * gain
    return coords

# 웹캠 연결
cap = cv2.VideoCapture(0)

# 거리 측정 임계값 (가정)
threshold_distance = 50

# YOLOv5 모델 불러오기
device_str = ''
device = select_device(device_str)
model_path = 'yolov5s.pt'  # 적절한 파일 경로로 수정
model = attempt_load(model_path)
stride = int(model.stride.max())

while True:
    _, frame = cap.read()

    # YOLOv5를 사용하여 객체 감지
    img = torch.from_numpy(frame).to(device)
    img = img.float() / 255.0
    img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)[0]

    # 객체 감지 후 바운딩 박스 그리기 및 거리 측정
    if pred is not None:
        img_shape = frame.shape
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img_shape).round()
        for box in pred:
            x1, y1, x2, y2, conf, cls = box.tolist()
            print(f"Class: {int(cls)}, Confidence: {conf}, BBox: ({x1}, {y1}, {x2}, {y2})")

            if int(cls) == 0:  # 예시: 사람 클래스가 0일 때
                for other_box in pred:
                    x1_other, y1_other, x2_other, y2_other, _, cls_other = other_box.tolist()

                    if int(cls_other) == 1:  # 예시: 난간 클래스가 1일 때
                        # 유클리드 거리 계산
                        distance = np.sqrt((x1 - x1_other)**2 + (y1 - y1_other)**2)

                        # 거리가 일정 임계값 이하이면 "기대는 행동" 출력
                        if distance < threshold_distance:
                            print("기대는 행동")

    # 바운딩 박스 그리기
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = model.module.draw_boxes(frame, pred[0]) if hasattr(model, 'module') else model.draw_boxes(frame, pred[0])

    # 결과 출력
    cv2.imshow('YOLOv5 Object Detection', frame)

    if cv2.waitKey(1) == 27:  # 'ESC' 키로 종료
        break

cap.release()
cv2.destroyAllWindows()

