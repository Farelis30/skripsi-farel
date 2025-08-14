import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_finger_states(landmarks):
    """
    Deteksi jari terbuka/tutup.
    Return [thumb, index, middle, ring, pinky] -> True=open, False=close
    """
    # ====== 1) DETEKSI JEMPOL ======
    # hitung jarak thumb_tip ke index_mcp vs thumb_ip ke index_mcp
    thumb_tip_x = landmarks[4].x
    thumb_ip_x  = landmarks[3].x
    index_mcp_x = landmarks[5].x

    # selisih jarak ke index_mcp
    tip_dist = abs(thumb_tip_x - index_mcp_x)
    ip_dist  = abs(thumb_ip_x  - index_mcp_x)

    # jempol dianggap terbuka kalau tip lebih jauh dari telapak
    thumb_open = tip_dist > ip_dist + 0.02  # margin supaya nggak terlalu sensitif

    # ====== 2) DETEKSI JARI LAIN (VERTIKAL) ======
    finger_tips = [8, 12, 16, 20]
    finger_mcp  = [5, 9, 13, 17]

    fingers_open = []
    for tip, mcp in zip(finger_tips, finger_mcp):
        fingers_open.append(landmarks[tip].y < landmarks[mcp].y - 0.02)  # tambahin margin biar stabil

    return [thumb_open] + fingers_open

def classify_gesture(states):
    """
    Klasifikasi gesture 4 kelas: Maju, Stop, Kiri, Kanan
    """
    # Semua jari terbuka
    if all(states):
        return "Maju"
    # Semua tertutup
    if not any(states):
        return "Stop"
    # Hanya jempol terbuka
    if states[0] and not any(states[1:]):
        return "Kiri"
    # Hanya kelingking terbuka
    if states[-1] and not any(states[:-1]):
        return "Kanan"
    return "Tidak dikenal"

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        h, w, _ = frame.shape
        gesture_text = "Tidak ada tangan"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Gambar landmark
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Ambil status jari
                states = get_finger_states(hand_landmarks.landmark)

                # Konversi ke string misal: T:Open, I:Closed ...
                names = ["Jempol", "Telunjuk", "Tengah", "Manis", "Kelingking"]
                state_text = ", ".join([f"{n}:{'Buka' if s else 'Tutup'}" for n, s in zip(names, states)])

                # Prediksi gesture
                gesture = classify_gesture(states)

                # Tampilkan di layar
                gesture_text = f"Prediksi: {gesture}"
                cv2.putText(frame, state_text, (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.putText(frame, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Gesture Debug", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC untuk keluar
            break

cap.release()
cv2.destroyAllWindows()
