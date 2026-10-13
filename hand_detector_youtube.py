import os
import cv2
import mediapipe as mp
import yt_dlp

# ==========================================================
# 📥 유튜브 스트림 링크 추출 함수 (수정됨)
# ==========================================================
def get_youtube_stream(url):
    ydl_opts = {
        'format': 'bv*+ba/b',
        'noplaylist': True,
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        video_url = info['url']
        return video_url

# 🎬 슬라이드에 나온 유튜브 쇼츠/영상 주소 설정
youtube_url = 'https://www.youtube.com/shorts/xNQ6ysIWpr8?feature=share'

print('📥 유튜브 스트림 링크를 추출 중입니다...')
video_stream_url = get_youtube_stream(youtube_url)
print('✅ 스트림 링크 획득 완료!')

# ==========================================================
# ✋ Hand Detection 초기화
# ==========================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================================
# 🎥 유튜브 스트림 연결
# ==========================================================
cap = cv2.VideoCapture(video_stream_url)

if not cap.isOpened():
    print('❌ 유튜브 영상을 연결할 수 없습니다.')
    exit()

print('▶️ 유튜브 스트리밍 영상 재생 시작 : ESC를 눌러 종료합니다.')

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print('⚠️ 프레임을 읽지 못했습니다. 스트리밍 연결을 확인하세요.')
        break

    # 좌우 반전 (셀카 뷰)
    image = cv2.flip(image, 1)

    # BGR -> RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 손 감지 수행
    results = hands.process(image_rgb)

    # 🎨 손 랜드마크 시각화
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

    # 결과 화면 출력
    cv2.imshow('MediaPipe Hand Detector [YouTube]', image)

    # ESC 키(27)를 누르면 종료
    if cv2.waitKey(5) & 0xFF == 27:
        print('👋 종료합니다.')
        break

# ==========================================================
# 🔚 자원 해제
# ==========================================================
cap.release()
cv2.destroyAllWindows()