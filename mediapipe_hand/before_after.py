import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Nama file output
original_path = "results/frame_original.jpg"
landmark_path = "results/frame_with_landmarks.jpg"

# Buka webcam
cap = cv2.VideoCapture(0)

print("✅ Tekan [SPACE] untuk ambil foto, [ESC] untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membaca webcam.")
        break

    # Tampilkan preview
    cv2.imshow("Preview Webcam", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC untuk keluar
        break
    elif key == 32:  # SPASI untuk ambil gambar
        # Simpan gambar original
        cv2.imwrite(original_path, frame)
        print(f"✅ Gambar asli disimpan sebagai {original_path}")

        # Proses MediaPipe untuk mendeteksi tangan
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7) as hands:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            # Jika ada tangan terdeteksi → gambar landmark
            frame_landmark = frame.copy()
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame_landmark, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Simpan gambar dengan landmark
        cv2.imwrite(landmark_path, frame_landmark)
        print(f"✅ Gambar dengan landmark disimpan sebagai {landmark_path}")

        # Gabungkan kedua gambar berdampingan
        combined = np.hstack((frame, frame_landmark))
        cv2.imshow("Before (kiri) & After (kanan)", combined)

cap.release()
cv2.destroyAllWindows()
