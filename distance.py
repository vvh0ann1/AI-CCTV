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
weights_path = 'C:\\Users\\rabbi_fxg6wgn\\Downloads\\yolov5-20240203T010646Z-001\\yolov5\\runs\\train\\all\\weights\\best.pt'  # 학습시킨 모델의 가중치 경로
device = select_device('')
model = attempt_load(weights_path, device=device)

model.eval()

# 이미지 로드 및 전처리
image_path = 'C:\\Users\\rabbi_fxg6wgn\\Downloads\\yolov5-20240203T010646Z-001\\n2.jpg'  # 대상 이미지 경로
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
    railing_boxes = [] 
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
        elif class_idx == 1:  # 난간 클래스
            label = "railing"
            color = (255, 0, 0)  # 파란색
            railing_boxes.append(bbox)
            print(f"{label} Bounding Box Coordinates: {bbox}")
        elif class_idx == 2:  # 폭행 클래스
            label = "Violence"
            color = (0, 0, 255)  # 빨간색
            violence_box = bbox
            
            # 바운딩 박스 좌표 출력
            print(f"{label} Bounding Box Coordinates: {bbox}")

            # 바운딩 박스 그리기
            # 바운딩 박스 그리기
            print(f"Drawing {label} bounding box at ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]}) with color {color}")
            cv2.rectangle(img_array, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(img_array, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

       # "person" 클래스에 대한 바운딩 박스를 그립니다.
    for person_box in person_boxes:
        label = "person"
        color = (0, 255, 0)  # 초록색
        print(f"Drawing {label} bounding box at ({person_box[0]}, {person_box[1]}) to ({person_box[2]}, {person_box[3]}) with color {color}")
        cv2.rectangle(img_array, (person_box[0], person_box[1]), (person_box[2], person_box[3]), color, 2)
        cv2.putText(img_array, label, (person_box[0], person_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # "rooftop" 클래스에 대한 바운딩 박스를 그립니다.
    for railing_box in railing_boxes:
        label = "railing"
        color = (255, 0, 0)  # 파란색
        print(f"Drawing {label} bounding box at ({railing_box[0]}, {railing_box[1]}) to ({railing_box[2]}, {railing_box[3]}) with color {color}")
        cv2.rectangle(img_array, (railing_box[0], railing_box[1]), (railing_box[2], railing_box[3]), color, 2)
        cv2.putText(img_array, label, (railing_box[0], railing_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # "railing" 클래스에 대한 모든 바운딩 박스를 그립니다.
    if len(railing_boxes) > 0:  # 난간이 감지된 경우에만 실행
        # 가장 먼 좌표를 찾기
        farthest_railing = min(railing_boxes, key=lambda box: box[2] * box[3])  # 기준: 바운딩 박스의 넓이 (가로 * 세로)
        
        # 사람 객체와 난간 객체의 거리 비교
        if person_boxes:
            person_center = [(person_box[0] + person_box[2]) / 2 for person_box in person_boxes], [(person_box[1] + person_box[3]) / 2 for person_box in person_boxes]
            railing_center = [(farthest_railing[0] + farthest_railing[2]) / 2, (farthest_railing[1] + farthest_railing[3]) / 2]

            # 거리 계산
            distances = [np.linalg.norm(np.array(person_center[i]) - np.array(railing_center)) for i in range(len(person_center))]
            min_distance = min(distances)
            print(f"최소 거리: {min_distance} 픽셀")

            # 일정 거리 이하일 때 알림 출력
            threshold_distance = 180  # 예시: 일정 거리
            if min_distance < threshold_distance:
                print(f"경고: 사람과 가장 먼 난간이 너무 가깝습니다! 거리: {min_distance} 픽셀")

    # 화면에 이미지 출력
    cv2.imshow("Result", img_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("객체가 감지되지 않았습니다.")

  