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

# 영상 로드 및 전처리
video_path = 'C:\\Users\\rabbi_fxg6wgn\\Downloads\\yolo_python\\1.mp4'  # 대상 영상 경로
cap = cv2.VideoCapture(video_path)

# 영상의 프레임 크기 및 FPS 가져오기
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# 출력 영상 설정
output_path = 'C:\\Users\\rabbi_fxg6wgn\\Downloads\\yolo_python\\output.avi'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # 프레임을 모델이 예상하는 크기로 조정
    model_input_size = (640, 640)  # 모델의 입력 크기
    frame = cv2.resize(frame, model_input_size)

    # 프레임을 텐서로 변환
    frame_tensor = transforms.ToTensor()(frame).unsqueeze(0).to(device)

    # 객체 감지
    with torch.no_grad():
        results = model(frame_tensor)

    # 결과 후처리 및 바운딩 박스 출력
    results = non_max_suppression(results, conf_thres=0.5, iou_thres=0.5, max_det=100)[0]
    violence_detected = False  # 폭력 감지 여부 플래그

    if results is not None and len(results) >= 0:
        # PyTorch 텐서에서 NumPy 배열로 변환
        boxes = results.cpu().numpy()

        # 사람 객체, 난간 객체의 바운딩 박스 추출
        person_boxes = []
        railing_boxes = []  # 여러 난간 객체를 저장하기 위한 리스트
        violence_boxes = []  # 폭행 객체를 저장하기 위한 리스트

        for box in boxes:
            class_idx = int(box[5])
            bbox = box[:4].astype(int)

            if class_idx == 0:  # 사람 클래스
                person_boxes.append(bbox)
            elif class_idx == 1:  # 난간 클래스
                railing_boxes.append(bbox)
            elif class_idx == 2:  # 폭행 클래스
                violence_boxes.append(bbox)
                violence_detected = True  # 폭력 감지됨
                print(f"폭행 상황이 감지되었습니다.")

                # 바운딩 박스 그리기
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                cv2.putText(frame, "Violence", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # "railing" 클래스에 대한 모든 바운딩 박스를 그립니다.
        if len(railing_boxes) > 0 and len(person_boxes) > 0:  # 난간과 사람이 감지된 경우에만 실행
            max_IoP = 0

            for person_box in person_boxes:
                for railing_box in railing_boxes:
                    # 겹친 비율 계산
                    intersection = [max(railing_box[0], person_box[0]), max(railing_box[1], person_box[1]),
                                    min(railing_box[2], person_box[2]), min(railing_box[3], person_box[3])]

                    area_intersection = max(0, intersection[2] - intersection[0] + 1) * max(0, intersection[3] - intersection[1] + 1)

                    # 사람 객체의 전체 면적
                    area_person = (person_box[2] - person_box[0] + 1) * (person_box[3] - person_box[1] + 1)

                    # 겹친 비율 계산
                    corrent_IoP = (area_intersection / area_person) * 100

                    if corrent_IoP > max_IoP:
                        max_IoP = corrent_IoP

                    # Draw bounding boxes for Person and Railing
                    cv2.rectangle(frame, (person_box[0], person_box[1]), (person_box[2], person_box[3]), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (person_box[0], person_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.rectangle(frame, (railing_box[0], railing_box[1]), (railing_box[2], railing_box[3]), (255, 0, 0), 2)
                    cv2.putText(frame, "Railing", (railing_box[0], railing_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            if max_IoP > 70:  
                img_iop_detected = np.zeros((150, 300, 3), dtype=np.uint8)
                b, g, r, a = 255, 255, 255, 0
                fontpath = "fonts/gulim.ttc"
                font = ImageFont.truetype(fontpath, 20)
                img_pil_iop_detected = Image.fromarray(img_iop_detected)
                draw = ImageDraw.Draw(img_pil_iop_detected)
                draw.text((60, 70), "투신이 감지되었습니다 ", font=font, fill=(b, g, r, a))
                img_iop_detected = np.array(img_pil_iop_detected)
                cv2.imshow("IoP Detected", img_iop_detected) 
            print(f"IoP: {max_IoP}%")

    # 프레임에 IoP 값을 표시
    cv2.putText(frame, f"IoP: {max_IoP}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 화면에 영상 출력
    cv2.imshow("Video", frame)

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
    
    # 수정된 프레임을 출력 영상에 쓰기
    out.write(frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.release()
out.release()
cv2.destroyAllWindows()
