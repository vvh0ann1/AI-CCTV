import torch
import numpy as np
import cv2
import pathlib
from PIL import Image, ImageFont, ImageDraw
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
image_path = 'C:\\Users\\rabbi_fxg6wgn\\Downloads\\yolo_python\\n2.jpg'  # 대상 이미지 경로
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
violence_detected = False  # 폭력 감지 여부 플래그

if results is not None and len(results) >= 0:
    # PyTorch 텐서에서 NumPy 배열로 변환
    boxes = results.cpu().numpy()

    # 이미지를 NumPy 배열로 변환
    img_array = np.array(img)

    # 사람 객체, 난간 객체, 폭행 객체의 바운딩 박스 추출
    person_boxes = []
    railing_boxes = []  # 여러 난간 객체를 저장하기 위한 리스트
    violence_boxes = []  # 여러 폭행 객체를 저장하기 위한 리스트

    for box in boxes:
        class_idx = int(box[5])
        bbox = box[:4].astype(int)
        label = ""
        color = (0, 0, 0)  # 기본값

        if class_idx == 0:  # 사람 클래스
            label = "person"
            color = (0, 255, 0)  # 초록색
            person_boxes.append(bbox)

            # 바운딩 박스 그리기
            cv2.rectangle(img_array, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(img_array, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif class_idx == 1:  # 난간 클래스
            label = "railing"
            color = (255, 0, 0)  # 파란색
            railing_boxes.append(bbox)
            
            # 바운딩 박스 그리기
            cv2.rectangle(img_array, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(img_array, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif class_idx == 2:  # 폭행 클래스
            label = "violence"
            color = (0, 0, 255)  # 빨간색
            violence_boxes.append(bbox)
            violence_detected = True  # 폭력 감지됨
            print(f"폭행 상황이 감지되었습니다.")

            # 바운딩 박스 그리기
            cv2.rectangle(img_array, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(img_array, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # "railing" 클래스에 대한 모든 바운딩 박스를 그립니다.
    if len(railing_boxes) > 0:  # 난간이 감지된 경우에만 실행
        max_IoP = 0
        best_railing_box = None  # formatted_overlap_ratio를 초기화
        best_person_box = None  # IoP가 70% 이상인 경우 저장할 사람 객체의 바운딩 박스
        for railing_box in railing_boxes:
            # 겹친 비율 계산
            IoP = 0
            for person_box in person_boxes:
                intersection = [max(railing_box[0], person_box[0]), max(railing_box[1], person_box[1]),
                                min(railing_box[2], person_box[2]), min(railing_box[3], person_box[3])]
                
                area_intersection = max(0, intersection[2] - intersection[0] + 1) * max(0, intersection[3] - intersection[1] + 1)
                # 사람 객체의 전체 면적
                area_person = (person_box[2] - person_box[0] + 1) * (person_box[3] - person_box[1] + 1)
                # IoP 계산
                corrent_IoP = area_intersection / area_person
                if corrent_IoP > IoP:
                    IoP = corrent_IoP            
            # 현재 난간 객체가 가장 많이 겹친 경우 저장
            if IoP > max_IoP:
                max_IoP = IoP
            # IoP 값이 0.7 이상인 경우 새로운 OpenCV 창 열기
            if max_IoP > 0.7:
                img_iop_detected = np.zeros((150, 300, 3), dtype=np.uint8)
                b, g, r, a = 255, 255, 255, 0
                fontpath = "fonts/gulim.ttc"
                font = ImageFont.truetype(fontpath, 20)
                img_pil_iop_detected = Image.fromarray(img_iop_detected)
                draw = ImageDraw.Draw(img_pil_iop_detected)
                draw.text((60, 70), "투신이 감지되었습니다 ", font=font, fill=(b, g, r, a))
                img_iop_detected = np.array(img_pil_iop_detected)
                cv2.imshow("IoP Detected", img_iop_detected)

    cv2.imshow("Result", img_array)

    # 폭행이 감지된 경우 새로운 OpenCV 창 열기
    if violence_detected:
        img_violence_detected = np.zeros((150, 300, 3), dtype=np.uint8)
        b, g, r, a = 255, 255, 255, 0
        fontpath = "fonts/gulim.ttc"
        font = ImageFont.truetype(fontpath, 20)
        img_pil_violence_detected = Image.fromarray(img_violence_detected)
        draw = ImageDraw.Draw(img_pil_violence_detected)
        draw.text((60, 70), "폭행이 감지되었습니다 ", font=font, fill=(b, g, r, a))
        img_violence_detected = np.array(img_pil_violence_detected)
        cv2.imshow("Violence Detected", img_violence_detected)

cv2.waitKey()
cv2.destroyAllWindows()
