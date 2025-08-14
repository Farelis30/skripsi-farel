import cv2
import time
import csv
import numpy as np
import tensorflow as tf

# Load model MobileNetV2 yang sudah ditraining
model = tf.keras.models.load_model('content/models/best_hand_model.h5')

# Kelas gesture sesuai model kamu
GESTURES = ["Maju", "kanan", "Kiri", "stop"]
TRIALS_PER_GESTURE = 30

# Preprocess frame untuk MobileNetV2
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))  # MobileNetV2 default input size
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def measure_gesture(gesture_name, trials=30):
    cap = cv2.VideoCapture(0)
    results_list = []
    print(f"\n[INFO] Mulai pengujian untuk gesture: {gesture_name}")

    for i in range(trials):
        frame_count = 0
        start_time = time.time()
        latency_ms = 0

        # Ambil 1 detik data untuk hitung FPS
        while time.time() - start_time < 1.0:
            ret, frame = cap.read()
            if not ret:
                break

            # Hitung latensi prediksi
            t1 = time.time()
            processed = preprocess_frame(frame)
            _ = model.predict(processed, verbose=0)
            t2 = time.time()

            latency_ms = (t2 - t1) * 1000  # konversi ke ms
            frame_count += 1

        fps = frame_count / (time.time() - start_time)
        results_list.append((gesture_name, i+1, round(fps, 2), round(latency_ms, 2)))
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
