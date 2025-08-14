import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
mp_df = pd.read_csv("mediapipe_raw_data.csv")
mn_df = pd.read_csv("mobilenet_raw_data.csv")

# Tambahkan kolom model
mp_df["Model"] = "MediaPipe"
mn_df["Model"] = "MobileNetV2"

# Gabungkan data
df = pd.concat([mp_df, mn_df], ignore_index=True)

# Atur style seaborn
sns.set(style="whitegrid", font_scale=1.2)

# Grafik 1: FPS
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="Gesture", y="FPS", hue="Model", ci=None, estimator="mean")
plt.title("Perbandingan FPS per Gesture")
plt.ylabel("FPS Rata-rata")
plt.xlabel("Gesture")
plt.legend(title="Model")
plt.tight_layout()
plt.show()

# Grafik 2: Latency
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="Gesture", y="Latency_ms", hue="Model", ci=None, estimator="mean")
plt.title("Perbandingan Latency per Gesture")
plt.ylabel("Latency Rata-rata (ms)")
plt.xlabel("Gesture")
plt.legend(title="Model")
plt.tight_layout()
plt.show()
