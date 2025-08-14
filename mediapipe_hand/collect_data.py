import cv2
import time
import mediapipe as mp
import csv

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

GESTURES = ["Maju", "Stop", "Kiri", "Kanan"]
TRIALS_PER_GESTURE = 30

def measure_gesture(gesture_name, trials=30):
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    
    results_list = []
    print(f"\n[INFO] Mulai pengujian untuk gesture: {gesture_name}")

    for i in range(trials):
        frame_count = 0
        start_time = time.time()

        # Ambil 1 detik data untuk hitung FPS
        while time.time() - start_time < 1.0:
            ret, frame = cap.read()
            if not ret:
                break

            # Hitung latensi
            t1 = time.time()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)
            t2 = time.time()
            
            latency_ms = (t2 - t1) * 1000  # konversi ke ms
            frame_count += 1
            
        fps = frame_count / (time.time() - start_time)
        results_list.append((gesture_name, i+1, round(fps,2), round(latency_ms,2)))
        print(f"Percobaan {i+1}/{trials}: FPS={fps:.2f}, Latency={latency_ms:.2f} ms")

    cap.release()
    return results_list

if __name__ == "__main__":
    all_results = []
    for gesture in GESTURES:
        input(f"\nSiapkan posisi untuk gesture '{gesture}', lalu tekan ENTER untuk mulai...")
        data = measure_gesture(gesture, TRIALS_PER_GESTURE)
        all_results.extend(data)

    # Simpan ke CSV
    with open("results/raw_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Gesture", "Trial", "FPS", "Latency_ms"])
        writer.writerows(all_results)

    print("\n[INFO] Data selesai disimpan ke results/raw_data.csv")
