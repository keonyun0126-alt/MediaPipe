#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import mediapipe as mp

# =========================================
# 🧩 Mediapipe 모듈 초기화
# =========================================
# 1. 배경 분리 (Selfie Segmentation)
mp_selfie = mp.solutions.selfie_segmentation
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)

# 2. 얼굴 감지 박스 (Face Detection)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# 3. 눈·코·입·귀 랜드마크 (Face Mesh)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================================
# 📸 카메라 및 동영상 파일 연결
# =========================================
# cap = cv2.VideoCapture(0)  # 웹캠 사용 시
cap = cv2.VideoCapture("face.mp4")  # 동영상 파일 사용 시

print("📷 배경 블러(기본) + 얼굴 박스 + 눈·코·입·귀 점 찍기 통합 시작 — ESC를 눌러 종료합니다.")

# 💡 주요 랜드마크 인덱스 정의 (귀 부위 234, 454 추가)
LEFT_EYE = [33, 133, 160, 158, 130, 246]
RIGHT_EYE = [362, 263, 387, 385, 359, 466]
NOSE_TIP = [1, 4, 5, 195, 197]
LIPS = [61, 291, 0, 17, 78, 308, 13, 14]
EARS = [234, 454]  # 양쪽 귀 및 광대/옆얼굴 외곽 포인트

# =========================================
# 🎞️ 실시간 처리 루프
# =========================================
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("⚠️ 프레임을 읽지 못했습니다. 영상/카메라 연결을 확인하세요.")
        break

    # 좌우 반전 (셀카 모드)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # BGR → RGB 변환
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -----------------------------------------
    # ① 배경 분리 및 블러 합성 (기본 블러 적용: (31, 31))
    # -----------------------------------------
    result_seg = segmenter.process(rgb)
    mask = (result_seg.segmentation_mask > 0.5).astype(np.uint8)
    
    # 💡 배경 블러 강도 원복 ((31, 31))
    blur = cv2.GaussianBlur(frame, (31, 31), 0)
    
    output = frame * mask[..., None] + blur * (1 - mask[..., None])
    output = output.astype(np.uint8)

    # -----------------------------------------
    # ② 얼굴 인식 (바운딩 박스 및 신뢰도 표시)
    # -----------------------------------------
    result_face = face_detection.process(rgb)
    if result_face.detections:
        for detection in result_face.detections:
            box = detection.location_data.relative_bounding_box
            x = int(box.xmin * w)
            y = int(box.ymin * h)
            width = int(box.width * w)
            height = int(box.height * h)

            # 초록색 얼굴 박스 그리기
            cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            # 신뢰도 텍스트 표시
            confidence = detection.score[0]
            cv2.putText(output, f"Face: {confidence:.2f}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # -----------------------------------------
    # ③ 눈·코·입·귀 랜드마크 점 찍기
    # -----------------------------------------
    result_mesh = face_mesh.process(rgb)
    if result_mesh.multi_face_landmarks:
        for face_landmarks in result_mesh.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                px = int(landmark.x * w)
                py = int(landmark.y * h)

                # 부위별 색상 지정 (눈: 파랑, 코: 노랑, 입: 빨강, 귀: 보라색)
                if idx in LEFT_EYE or idx in RIGHT_EYE:
                    cv2.circle(output, (px, py), 2, (255, 0, 0), -1)
                elif idx in NOSE_TIP:
                    cv2.circle(output, (px, py), 3, (0, 255, 255), -1)
                elif idx in LIPS:
                    cv2.circle(output, (px, py), 2, (0, 0, 255), -1)
                elif idx in EARS:
                    cv2.circle(output, (px, py), 5, (255, 0, 255), -1)  # 귀: 보라색 큰 점

    # 결과 화면 출력
    cv2.imshow("✨ MediaPipe All-in-One Lab (ESC to quit)", output)

    # ESC 키(ASCII 코드 27)를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        print("👋 종료합니다.")
        break

# =========================================
# 🔚 자원 해제
# =========================================
cap.release()
cv2.destroyAllWindows()
segmenter.close()
face_detection.close()
face_mesh.close()