import torch
import cv2
import pathlib
from torchvision import transforms
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

# YOLOv5 모델 로드
weights_path = 'C:\\Users\\rabbi_fxg6wgn\\Downloads\\yolov5-20240203T010646Z-001\\yolov5\\runs\\train\\exp\\weights\\best.pt'
device = select_device('')
model = attempt_load(weights_path, device=device)
model.eval()

# 웹캠 초기화
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠, 웹캠 번호에 따라 조절 가능

while True:
    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        continue  # 프레임을 읽지 못한 경우, 다음 루프로 넘어감

    # 프레임 전처리
    img_tensor = transforms.ToTensor()(frame).unsqueeze(0).to(device)

    # 객체 감지
    with torch.no_grad():
        try:
            results = model(img_tensor)
        except Exception as e:
            print(f"Error during inference: {e}")
            continue

    # 결과 후처리 및 바운딩 박스 출력
    results = non_max_suppression(results, conf_thres=0.5, iou_thres=0.5)[0]

    if results is not None and len(results) >= 1:
        # PyTorch 텐서에서 NumPy 배열로 변환
        boxes = results.cpu().numpy()

        for box in boxes:
            class_idx = int(box[5])
            bbox = box[:4].astype(int)
            label = ""
            color = (0, 0, 0)  # 기본값

            if class_idx == 1:  # 사람 클래스
                label = "Person"
                color = (0, 255, 0)  # 초록색
            elif class_idx == 2:  # 난간 클래스
                label = "Handrail"
                color = (255, 0, 0)  # 파란색
            elif class_idx == 0:  # 폭행 클래스
                label = "Violence"
                color = (0, 0, 255)  # 빨간색

            # 바운딩 박스 좌표 출력
            print(f"{label} Bounding Box Coordinates: {bbox}")

            # 바운딩 박스 그리기
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 화면에 이미지 출력
        cv2.imshow("Webcam Object Detection", frame)
        cv2.imshow("Webcam", frame)  # 추가된 부분

    else:
        print("No objects detected.")
        cv2.imshow("Webcam", frame)  # 추가된 부분

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 윈도우 종료
cap.release()
cv2.destroyAllWindows()
