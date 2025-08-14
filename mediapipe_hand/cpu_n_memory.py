import cv2
import mediapipe as mp
import psutil
import time
import matplotlib.pyplot as plt
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_finger_states(landmarks):
    """Deteksi jari terbuka/tutup."""
    # ====== 1) DETEKSI JEMPOL ======
    thumb_tip_x = landmarks[4].x
    thumb_ip_x  = landmarks[3].x
    index_mcp_x = landmarks[5].x

    tip_dist = abs(thumb_tip_x - index_mcp_x)
    ip_dist  = abs(thumb_ip_x  - index_mcp_x)
    thumb_open = tip_dist > ip_dist + 0.02

    # ====== 2) DETEKSI JARI LAIN ======
    finger_tips = [8, 12, 16, 20]
    finger_mcp  = [5, 9, 13, 17]

    fingers_open = []
    for tip, mcp in zip(finger_tips, finger_mcp):
        fingers_open.append(landmarks[tip].y < landmarks[mcp].y - 0.02)

    return [thumb_open] + fingers_open

def classify_gesture(states):
    """Klasifikasi gesture."""
    if all(states):
        return "Maju"
    if not any(states):
        return "Stop"
    if states[0] and not any(states[1:]):
        return "Kiri"
    if states[-1] and not any(states[:-1]):
        return "Kanan"
    return "Tidak dikenal"

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak bisa membuka kamera.")
        return

    os.makedirs("results", exist_ok=True)

    cpu_usage = []
    mem_usage = []
    timestamps = []
    start_time = time.time()

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Catat waktu dan resource usage
            elapsed_time = time.time() - start_time
            timestamps.append(elapsed_time)
            cpu_usage.append(psutil.cpu_percent(interval=None))
            mem_usage.append(psutil.virtual_memory().percent)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            h, w, _ = frame.shape
            gesture_text = "Tidak ada tangan"

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    states = get_finger_states(hand_landmarks.landmark)
                    names = ["Jempol", "Telunjuk", "Tengah", "Manis", "Kelingking"]
                    state_text = ", ".join([f"{n}:{'Buka' if s else 'Tutup'}" for n, s in zip(names, states)])
                    gesture = classify_gesture(states)
                    gesture_text = f"Prediksi: {gesture}"
                    cv2.putText(frame, state_text, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            cv2.putText(frame, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Gesture Debug (CPU & Memory Test)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()

    # Simpan grafik CPU & Memory
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, cpu_usage, label="CPU Usage (%)", color='r')
    plt.plot(timestamps, mem_usage, label="Memory Usage (%)", color='b')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Usage (%)")
    plt.title("CPU & Memory Usage During MediaPipe Gesture Detection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/cpu_memory_usage_mediapipe.png")
    print("[INFO] Grafik CPU & Memory usage disimpan di results/cpu_memory_usage_mediapipe.png")

if __name__ == "__main__":
    main()
