# Real-time Hand Detection dengan OpenCV
# Jalankan kode ini di lokal Anda

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

def preprocess_frame(frame):
    """Preprocess frame untuk prediksi"""
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hands
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        # Get hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get bounding box
        h, w, _ = frame.shape
        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding
        padding_x = int((x_max - x_min) * 0.2)
        padding_y = int((y_max - y_min) * 0.2)
        
        x_min = max(0, x_min - padding_x)
        x_max = min(w, x_max + padding_x)
        y_min = max(0, y_min - padding_y)
        y_max = min(h, y_max + padding_y)
        
        # Crop and resize
        cropped = frame[y_min:y_max, x_min:x_max]
        if cropped.size > 0:
            processed = cv2.resize(cropped, (224, 224))
            processed = processed.astype(np.float32) / 255.0
            processed = np.expand_dims(processed, axis=0)
            
            return processed, (x_min, y_min, x_max, y_max)
    
    return None, None

def main():
    """Main function untuk real-time detection"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Preprocess frame
        processed_frame, bbox = preprocess_frame(frame)
        
        if processed_frame is not None:
            # Make prediction
            prediction = model.predict(processed_frame, verbose=0)
            class_idx = np.argmax(prediction)
            confidence = prediction[0][class_idx]
            class_name = CLASS_NAMES[class_idx]
            
            # Draw bounding box only
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Display prediction
            text = f'{class_name}: {confidence:.2f}'
            cv2.putText(frame, text, (x_min, y_min-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Hand Gesture Detection', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
