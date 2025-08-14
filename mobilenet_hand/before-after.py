import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

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

# Nama file output
before_path = "results/frame_before.jpg"
after_path = "results/frame_after.jpg"

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
        print("❌ Error: Cannot open camera")
        return
    
    print("✅ Tekan [SPACE] untuk ambil foto, [ESC] untuk keluar.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # mirror effect
        frame_before = frame.copy()  # simpan versi sebelum edit
        frame_after = frame.copy()   # versi yang akan digambar kotak
        
        processed_frame, bbox = preprocess_frame(frame_after)
        
        if processed_frame is not None:
            prediction = model.predict(processed_frame, verbose=0)
            class_idx = np.argmax(prediction)
            confidence = prediction[0][class_idx]
            class_name = CLASS_NAMES[class_idx]
            
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame_after, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            text = f'{class_name}: {confidence:.2f}'
            cv2.putText(frame_after, text, (x_min, y_min-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow("Preview", frame_after)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC untuk keluar
            break
        elif key == 32:  # SPACE untuk simpan before & after
            cv2.imwrite(before_path, frame_before)
            cv2.imwrite(after_path, frame_after)
            print(f"📸 Before disimpan: {before_path}")
            print(f"📸 After disimpan: {after_path}")
            
            # Gabungkan before-after untuk ditampilkan
            combined = np.hstack((frame_before, frame_after))
            cv2.imshow("Before (Kiri) & After (Kanan)", combined)
            cv2.waitKey(0)  # tunggu tombol apapun untuk lanjut
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
