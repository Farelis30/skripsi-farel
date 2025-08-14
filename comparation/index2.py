import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data akurasi
mp_gesture = pd.read_csv("mediapipe_gesture.csv")
mn_gesture = pd.read_csv("mobilenet_gesture.csv")

# Hitung akurasi per gesture
def compute_accuracy(df):
    return df.groupby("TrueLabel").apply(lambda x: (x["TrueLabel"]==x["PredictedLabel"]).mean()).to_dict()

mp_acc = compute_accuracy(mp_gesture)
mn_acc = compute_accuracy(mn_gesture)

# Gesture urutan
gestures = ["Maju","Stop","Kiri","Kanan"]

# Buat bar chart
x = np.arange(len(gestures))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
ax.bar(x - width/2, [mp_acc[g]*100 for g in gestures], width, label="MediaPipe", color="skyblue")
ax.bar(x + width/2, [mn_acc[g]*100 for g in gestures], width, label="MobileNetV2", color="steelblue")

ax.set_ylabel("Akurasi (%)")
ax.set_xlabel("Gesture")
ax.set_title("Perbandingan Akurasi per Gesture")
ax.set_xticks(x)
ax.set_xticklabels(gestures)
ax.legend()

# Tambah label persentase di atas bar
for i in range(len(gestures)):
    ax.text(i - width/2, mp_acc[gestures[i]]*100 + 1, f"{mp_acc[gestures[i]]*100:.1f}%", ha="center")
    ax.text(i + width/2, mn_acc[gestures[i]]*100 + 1, f"{mn_acc[gestures[i]]*100:.1f}%", ha="center")

plt.tight_layout()
plt.show()
