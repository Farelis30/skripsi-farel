import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# === Setup MediaPipe ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# === Konfigurasi ===
GESTURES = ["Maju", "Stop", "Kiri", "Kanan"]
TRIALS_PER_GESTURE = 50  # percobaan per gesture

# Pastikan folder results ada
os.makedirs("results", exist_ok=True)

# ==========================
# 1) DETEKSI STATUS JARI
# ==========================
def get_finger_states(landmarks):
    """
    Deteksi jari terbuka/tutup.
    Output: [thumb, index, middle, ring, pinky] -> True=open, False=close
    """
    # ====== JEMPOL ======
    thumb_tip_x = landmarks[4].x
    thumb_ip_x  = landmarks[3].x
    index_mcp_x = landmarks[5].x

    # bandingkan jarak ke telapak (index_mcp)
    tip_dist = abs(thumb_tip_x - index_mcp_x)
    ip_dist  = abs(thumb_ip_x  - index_mcp_x)

    # jempol dianggap terbuka kalau tip lebih jauh dari telapak (ada margin)
    thumb_open = tip_dist > ip_dist + 0.02  

    # ====== 4 JARI LAIN ======
    finger_tips = [8, 12, 16, 20]
    finger_mcp  = [5, 9, 13, 17]

    fingers_open = []
    for tip, mcp in zip(finger_tips, finger_mcp):
        # terbuka kalau ujung lebih tinggi (lebih kecil Y)
        fingers_open.append(landmarks[tip].y < landmarks[mcp].y - 0.02)

    return [thumb_open] + fingers_open

# ==========================
# 2) KLASIFIKASI GESTURE
# ==========================
def classify_4gestures(landmarks):
    states = get_finger_states(landmarks)  # [thumb, index, middle, ring, pinky]

    # Semua jari terbuka => MAJU
    if all(states):
        return "Maju"
    # Semua tertutup => STOP
    elif not any(states):
        return "Stop"
    # Hanya jempol terbuka => KIRI
    elif states[0] and not any(states[1:]):
        return "Kiri"
    # Hanya kelingking terbuka => KANAN
    elif states[-1] and not any(states[:-1]):
        return "Kanan"
    else:
        # Kalau nggak cocok => fallback Stop
        return "Stop"

# ==========================
# 3) AMBIL DATA PER GESTURE
# ==========================
def capture_gesture_data(true_label, trials=20):
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    results_list = []

    print(f"\n[INFO] Mulai pengujian untuk gesture: {true_label}")

    count = 0
    while count < trials:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                predicted = classify_4gestures(hand_landmarks.landmark)
                
                results_list.append((true_label, predicted))
                count += 1
                print(f"[{count}/{trials}] True={true_label}, Predicted={predicted}")
                time.sleep(0.3)  # jeda 300ms
                break

        cv2.imshow("Gesture Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC keluar
            break

    cap.release()
    cv2.destroyAllWindows()
    return results_list

# ==========================
# 4) MAIN PROGRAM
# ==========================
if __name__ == "__main__":
    all_results = []

    # Ambil data tiap gesture
    for gesture in GESTURES:
        input(f"\nSiapkan gesture '{gesture}', lalu tekan ENTER untuk mulai...")
        data = capture_gesture_data(gesture, TRIALS_PER_GESTURE)
        all_results.extend(data)

    # Simpan raw hasil ke CSV
    raw_csv_path = "results/gesture_test_results_4class.csv"
    with open(raw_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TrueLabel", "PredictedLabel"])
        writer.writerows(all_results)
    print(f"\n[INFO] Data disimpan ke {raw_csv_path}")

    # ======================
    # Analisis Akurasi
    # ======================
    y_true = [row[0] for row in all_results]
    y_pred = [row[1] for row in all_results]

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=GESTURES)
    print("\nConfusion Matrix:\n", cm)

    # Precision, Recall, F1 per gesture
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

    # Simpan CSV metrics
    metrics_csv_path = "gesture/gesture_metrics_4class.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"[INFO] Precision/Recall/F1 disimpan ke {metrics_csv_path}")

    # Classification report text lengkap
    report = classification_report(y_true, y_pred, target_names=GESTURES, digits=3)
    with open("gesture/classification_report_4class.txt", "w") as f:
        f.write(report)

    # Simpan confusion matrix CSV
    np.savetxt("gesture/confusion_matrix_4class.csv", cm, delimiter=",", fmt="%d")

    # Heatmap confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=GESTURES, yticklabels=GESTURES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix 4 Gesture")
    plt.tight_layout()
    plt.savefig("gesture/confusion_matrix_4class.png", dpi=150)
    plt.show()

    print("\n[INFO] Semua hasil (CSV, PNG, TXT) disimpan di folder gesture/")
    print("\n[INFO] Selesai!")