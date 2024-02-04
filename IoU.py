import torch
import numpy as np
import cv2
import pathlib
from PIL import Image
from torchvision import transforms
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

# YOLOv5 모델 로드
weights_path = 'C:\\Users\\rabbi_fxg6wgn\\Downloads\\yolov5-20240203T010646Z-001\\yolov5\\runs\\train\\result\\best.pt'  # 학습시킨 모델의 가중치 경로
device = select_device('')
model = attempt_load(weights_path, device=device)

model.eval()

# 이미지 로드 및 전처리
image_path = 'C:\\Users\\rabbi_fxg6wgn\\Downloads\\yolov5-20240203T010646Z-001\\n9.jpg'  # 대상 이미지 경로
img = Image.open(image_path)
img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

# 객체 감지
with torch.no_grad():
    results = model(img_tensor)

# 결과 후처리 및 바운딩 박스 출력
results = non_max_suppression(results, conf_thres=0.5, iou_thres=0.5, max_det=100)[0]

if results is not None and len(results) >= 0:
    # PyTorch 텐서에서 NumPy 배열로 변환
    boxes = results.cpu().numpy()

    # 이미지를 NumPy 배열로 변환
    img_array = np.array(img)

    # 사람 객체, 난간 객체, 폭행 객체의 바운딩 박스 추출
    person_boxes = []
    railing_boxes = []  # 여러 난간 객체를 저장하기 위한 리스트
    violence_box = None

    for box in boxes:
        class_idx = int(box[5])
        bbox = box[:4].astype(int)
        label = ""
        color = (0, 0, 0)  # 기본값

        if class_idx == 0:  # 사람 클래스
            label = "person"
            color = (0, 255, 0)  # 초록색
            person_boxes.append(bbox)
            print(f"{label} Bounding Box Coordinates: {bbox}")

            # 바운딩 박스 그리기
            print(f"Drawing {label} bounding box at ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]}) with color {color}")
            cv2.rectangle(img_array, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(img_array, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif class_idx == 1:  # 난간 클래스
            label = "railing"
            color = (255, 0, 0)  # 파란색
            railing_boxes.append(bbox)
            print(f"{label} Bounding Box Coordinates: {bbox}")

            # 바운딩 박스 그리기
            print(f"Drawing {label} bounding box at ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]}) with color {color}")
            cv2.rectangle(img_array, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(img_array, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # "railing" 클래스에 대한 모든 바운딩 박스를 그립니다.
    if len(railing_boxes) > 0:  # 난간이 감지된 경우에만 실행
        # 가장 큰 난간 찾기
        farthest_railing = max(railing_boxes, key=lambda box: box[2] * box[3])  # 기준: 바운딩 박스의 넓이 (가로 * 세로)
        for railing_box in railing_boxes:
            label = "railing"
            color = (255, 0, 0)  # 파란색
            print(f"Drawing {label} bounding box at ({railing_box[0]}, {railing_box[1]}) to ({railing_box[2]}, {railing_box[3]}) with color {color}")
            cv2.rectangle(img_array, (railing_box[0], railing_box[1]), (railing_box[2], railing_box[3]), color, 2)
            cv2.putText(img_array, label, (railing_box[0], railing_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # IoU를 이용하여 가장 큰 난간과의 겹침을 확인
        max_iou = 0
        for person_box in person_boxes:
            intersection = [max(farthest_railing[0], person_box[0]), max(farthest_railing[1], person_box[1]),
                            min(farthest_railing[2], person_box[2]), min(farthest_railing[3], person_box[3])]

            area_intersection = max(0, intersection[2] - intersection[0] + 1) * max(0, intersection[3] - intersection[1] + 1)
            area_union = (farthest_railing[2] - farthest_railing[0] + 1) * (farthest_railing[3] - farthest_railing[1] + 1) + \
                        (person_box[2] - person_box[0] + 1) * (person_box[3] - person_box[1] + 1) - area_intersection

            iou = area_intersection / area_union if area_union > 0 else 0

            if iou > max_iou:
                max_iou = iou

        # 난간 객체의 IoU가 70% 이상일 때 알림 출력
        threshold_iou = 0.09  # 예시: 난간 객체의 일정 IoU
        if max_iou > threshold_iou:
            print(f"경고: 사람과 가장 큰 난간이 너무 많이 겹쳤습니다! IoU: {max_iou}")
        else:
            print(f"가장 큰 IoU: {max_iou}")

    # 화면에 이미지 출력
    cv2.imshow("Result", img_array)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print("객체가 감지되지 않았습니다.")
