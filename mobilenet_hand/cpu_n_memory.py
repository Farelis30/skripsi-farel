import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import psutil
import time
import matplotlib.pyplot as plt
import os

# Load model yang sudah ditraining
model = tf.keras.models.load_model('content/models/best_hand_model.h5')

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    max_num_hands=1
)

# Class names
CLASS_NAMES = ['maju', 'kanan', 'kiri', 'stop']

def preprocess_frame(frame):
    """Preprocess frame untuk prediksi"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        padding_x = int((x_max - x_min) * 0.2)
        padding_y = int((y_max - y_min) * 0.2)
        
        x_min = max(0, x_min - padding_x)
        x_max = min(w, x_max + padding_x)
        y_min = max(0, y_min - padding_y)
        y_max = min(h, y_max + padding_y)
        
        cropped = frame[y_min:y_max, x_min:x_max]
        if cropped.size > 0:
            processed = cv2.resize(cropped, (224, 224))
            processed = processed.astype(np.float32) / 255.0
            processed = np.expand_dims(processed, axis=0)
            return processed, (x_min, y_min, x_max, y_max)
    
    return None, None

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    # Buat folder results jika belum ada
    os.makedirs("results", exist_ok=True)

    # List untuk menyimpan data monitoring
    cpu_usage = []
    mem_usage = []
    timestamps = []

    print("Press 'q' to quit")

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Catat waktu sekarang untuk grafik
        elapsed_time = time.time() - start_time
        timestamps.append(elapsed_time)

        # Ambil penggunaan CPU dan Memory
        cpu_percent = psutil.cpu_percent(interval=None)
        mem_percent = psutil.virtual_memory().percent
        cpu_usage.append(cpu_percent)
        mem_usage.append(mem_percent)

        processed_frame, bbox = preprocess_frame(frame)

        if processed_frame is not None:
            prediction = model.predict(processed_frame, verbose=0)
            class_idx = np.argmax(prediction)
            confidence = prediction[0][class_idx]
            class_name = CLASS_NAMES[class_idx]

            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            text = f'{class_name}: {confidence:.2f}'
            cv2.putText(frame, text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Hand Gesture Detection (CPU & Memory Test)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Simpan grafik CPU & Memory usage
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, cpu_usage, label="CPU Usage (%)", color='r')
    plt.plot(timestamps, mem_usage, label="Memory Usage (%)", color='b')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Usage (%)")
    plt.title("CPU & Memory Usage During Hand Detection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/cpu_memory_usage.png")
    print("[INFO] Grafik CPU & Memory usage disimpan di results/cpu_memory_usage.png")

if __name__ == "__main__":
    main()
