# 🚀 MediaPipe All-in-One Computer Vision Project

> **간단한 소개**: 이 저장소는 Google MediaPipe와 OpenCV를 활용하여 인공지능 기반 컴퓨터 비전 기술(손/얼굴 감지, 배경 분리)을 단계별로 학습하고 통합한 실습 프로젝트입니다.

---

## 📑 Table of Contents
• [프로젝트 개요 및 주요 기능](#-프로젝트-개요-및-주요-기능)
• [프로젝트 구조 (Tree)](#-프로젝트-구조-tree)
• [실행 방법 및 명령어](#-실행-방법-및-명령어)
• [핵심 코드 스니펫](#-핵심-코드-스니펫)
• [활용 기술 및 참고 링크](#-활용-기술-및-참고-링크)

---

## 📌 프로젝트 개요 및 주요 기능

### 1. ✋ Hand Detection (손 랜드마크 추적)
- 로컬 동영상 파일(`hand.mp4`) 및 웹캠을 연동하여 손가락의 21개 키포인트 검출
- `yt-dlp` 라이브러리를 활용한 유튜브 스트리밍 영상 링크 실시간 연동 (`hand_detector_youtube.py`)

### 2. 🪞 Selfie Segmentation (실시간 배경 분리 및 블러)
- 인물(Foreground)과 배경(Background)을 실시간으로 분리
- 가우시안 블러(`cv2.GaussianBlur`)를 적용하여 화상회의처럼 배경을 흐리게 처리하는 효과 구현

### 3. 👤 Face Detection & Face Mesh (얼굴 인식 및 랜드마크)
- 얼굴 영역 바운딩 박스 및 신뢰도(Confidence Score) 출력
- 얼굴의 주요 부위인 **눈(파란색), 코(노란색), 입술(빨간색), 귀(보라색)** 위치에 정밀한 포인트(점) 시각화 적용

---

## 📂 프로젝트 구조 (Tree)

```text
MediaPipe/
├── .venv/                      # 파이썬 가상환경 폴더 (Git 업로드 제외)
├── .gitignore                  # Git 버전 관리 예외 설정 (.venv/, *.mp4 등)
├── README.md                   # 프로젝트 설명서
├── requirements.txt            # 필수 라이브러리 목록 파일
├── face.mp4                    # 얼굴 인식 및 배경 분리 테스트 동영상
├── hand.mp4                    # 손 인식 테스트 동영상
├── hand_detector.py            # 기본 손 랜드마크 감지 스크립트
├── hand_detector_youtube.py    # 유튜브 스트림 연동 손 감지 스크립트
└── selfie_segmentation.py      # 배경 블러 + 얼굴 박스 + 눈·코·입·귀 랜드마크 통합 스크립트