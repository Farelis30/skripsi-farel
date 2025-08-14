import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# === Load Model CNN MobileNetV2 ===
MODEL_PATH = "content/models/best_hand_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# === Setup MediaPipe Hands ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# === Konfigurasi ===
GESTURES = ["maju", "kanan", "kiri", "stop"]  # Sesuai urutan training model
TRIALS_PER_GESTURE = 50

# Pastikan folder results ada
os.makedirs("results", exist_ok=True)

# === Fungsi Preprocessing Frame untuk Model ===
def preprocess_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        h, w, _ = frame.shape
        hand_landmarks = results.multi_hand_landmarks[0]
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

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

# === Fungsi Prediksi Gesture ===
def classify_gesture(frame):
    processed_frame, _ = preprocess_frame(frame)
    if processed_frame is not None:
        prediction = model.predict(processed_frame, verbose=0)
        class_idx = np.argmax(prediction)
        return GESTURES[class_idx]
    return "stop"  # fallback

# === Capture Data untuk Evaluasi ===
def capture_gesture_data(true_label, trials=20):
    cap = cv2.VideoCapture(0)
    results_list = []

    print(f"\n[INFO] Mulai pengujian untuk gesture: {true_label}")

    count = 0
    while count < trials:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        predicted = classify_gesture(frame)

        # Simpan hasil prediksi
        results_list.append((true_label, predicted))
        count += 1
        print(f"[{count}/{trials}] True={true_label}, Predicted={predicted}")
        time.sleep(0.3)  # jeda 300ms

        cv2.imshow("Gesture Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC keluar
            break

    cap.release()
    cv2.destroyAllWindows()
    return results_list

# === Main Program ===
if __name__ == "__main__":
    all_results = []

    for gesture in GESTURES:
        input(f"\nSiapkan gesture '{gesture}', lalu tekan ENTER untuk mulai...")
        data = capture_gesture_data(gesture, TRIALS_PER_GESTURE)
        all_results.extend(data)

    # Simpan hasil raw
    raw_csv_path = "results/gesture_test_results_4class.csv"
    with open(raw_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TrueLabel", "PredictedLabel"])
        writer.writerows(all_results)
    print(f"\n[INFO] Data disimpan ke {raw_csv_path}")

    # Analisis
    y_true = [row[0] for row in all_results]
    y_pred = [row[1] for row in all_results]

    cm = confusion_matrix(y_true, y_pred, labels=GESTURES)
    print("\nConfusion Matrix:\n", cm)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=GESTURES, zero_division=0
    )

    metrics_df = pd.DataFrame({
        "Gesture": GESTURES,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Support": support
    })

    print("\n=== Precision, Recall, F1 per Gesture ===")
    print(metrics_df.to_string(index=False))

    metrics_csv_path = "results/gesture_metrics_4class.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    report = classification_report(y_true, y_pred, target_names=GESTURES, digits=3)
    with open("results/classification_report_4class.txt", "w") as f:
        f.write(report)

    np.savetxt("results/confusion_matrix_4class.csv", cm, delimiter=",", fmt="%d")

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=GESTURES, yticklabels=GESTURES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix 4 Gesture")
    plt.tight_layout()
    plt.savefig("results/confusion-matrix-mobilenet.png", dpi=150)
    plt.show()

    print("\n[INFO] Semua hasil (CSV, PNG, TXT) disimpan di folder results/")
    print("\n[INFO] Selesai!")
