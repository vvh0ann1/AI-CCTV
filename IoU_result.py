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
weights_path = 'C:\\Users\\rabbi_fxg6wgn\\Downloads\\yolo_python\\yolov5\\runs\\train\\result\\best.pt'  # 학습시킨 모델의 가중치 경로
device = select_device('')
model = attempt_load(weights_path, device=device)

model.eval()

# 이미지 로드 및 전처리
image_path = 'C:\\Users\\rabbi_fxg6wgn\\Downloads\\yolo_python\\n11.jpg'  # 대상 이미지 경로
img = Image.open(image_path)

# 이미지를 모델이 예상하는 크기로 조정
model_input_size = (640, 640)  # 모델의 입력 크기
img = img.resize(model_input_size)

# 이미지를 텐서로 변환
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
        max_overlap_ratio = 0
        best_railing_box = None

        for railing_box in railing_boxes:
            label = "railing"
            color = (255, 0, 0)  # 파란색
            print(f"Drawing {label} bounding box at ({railing_box[0]}, {railing_box[1]}) to ({railing_box[2]}, {railing_box[3]}) with color {color}")
            cv2.rectangle(img_array, (railing_box[0], railing_box[1]), (railing_box[2], railing_box[3]), color, 2)
            cv2.putText(img_array, label, (railing_box[0], railing_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 겹친 비율 계산
            overlap_ratio = 0
            for person_box in person_boxes:
                intersection = [max(railing_box[0], person_box[0]), max(railing_box[1], person_box[1]),
                                min(railing_box[2], person_box[2]), min(railing_box[3], person_box[3])]

                area_intersection = max(0, intersection[2] - intersection[0] + 1) * max(0, intersection[3] - intersection[1] + 1)

                # 사람 객체의 전체 면적
                area_person = (person_box[2] - person_box[0] + 1) * (person_box[3] - person_box[1] + 1)

                # 겹친 비율 계산
                current_overlap_ratio = (area_intersection / area_person) * 100

                if current_overlap_ratio > overlap_ratio:
                    overlap_ratio = current_overlap_ratio

            # 현재 난간 객체가 가장 많이 겹친 경우 저장
            if overlap_ratio > max_overlap_ratio:
                max_overlap_ratio = overlap_ratio
                best_railing_box = railing_box

        # 가장 많이 겹친 난간 객체에 대한 알림 출력
        if best_railing_box is not None and max_overlap_ratio > 70:
            print(f"경고: 사람과 가장 많이 겹친 난간이 있습니다! 겹친 비율: {max_overlap_ratio}%")

        # 가장 큰 비율 출력
        print(f"가장 큰 겹친 비율: {max_overlap_ratio}")

    # 화면에 이미지 출력
    cv2.imshow("Result", img_array)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print("객체가 감지되지 않았습니다.")
