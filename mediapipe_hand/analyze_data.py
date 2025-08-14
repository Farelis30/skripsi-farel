import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Baca data mentah
df = pd.read_csv("results/raw_data.csv")

# Hitung rata-rata per gesture
summary = df.groupby("Gesture").agg(
    avg_fps=("FPS", "mean"),
    avg_latency=("Latency_ms", "mean")
).reset_index()

# Simpan summary ke CSV
summary.to_csv("results/summary.csv", index=False)
print(summary)

# Buat grafik
plt.figure(figsize=(8,5))
sns.set_style("whitegrid")

bar_width = 0.4
x = range(len(summary))

plt.bar(x, summary["avg_fps"], width=bar_width, label="FPS", color="skyblue")
plt.bar([p + bar_width for p in x], summary["avg_latency"], width=bar_width, label="Latency (ms)", color="orange")

plt.xticks([p + bar_width/2 for p in x], summary["Gesture"])
plt.ylabel("FPS / Latency (ms)")
plt.title("FPS dan Latency per Gesture (MediaPipe)")
plt.legend()
plt.tight_layout()
plt.savefig("results/fps_latency_chart.png", dpi=150)
plt.show()
