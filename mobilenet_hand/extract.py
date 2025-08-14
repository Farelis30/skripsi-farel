import cv2
import os
import numpy as np
from pathlib import Path

def extract_frames_from_video(video_path, output_folder, target_frames=100, resize_to=(224, 224)):
    """
    Ekstrak frame dari video dan simpan sebagai PNG
    
    Args:
        video_path (str): Path ke file video
        output_folder (str): Folder output untuk menyimpan PNG
        target_frames (int): Jumlah frame yang ingin diekstrak
        resize_to (tuple): Ukuran gambar output (width, height)
    """
    
    # Buka video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Tidak dapat membuka video {video_path}")
        return False
    
    # Dapatkan informasi video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Durasi: {duration:.2f} detik")
    
    # Buat folder output jika belum ada
    os.makedirs(output_folder, exist_ok=True)
    
    # Hitung interval frame untuk mendapatkan target_frames
    if total_frames <= target_frames:
        # Jika total frame kurang dari target, ambil semua frame
        frame_interval = 1
        frames_to_extract = total_frames
    else:
        # Hitung interval untuk mendapatkan target_frames secara merata
        frame_interval = total_frames // target_frames
        frames_to_extract = target_frames
    
    print(f"Akan mengekstrak {frames_to_extract} frame dengan interval {frame_interval}")
    
    extracted_count = 0
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Ekstrak frame sesuai interval
        if frame_number % frame_interval == 0 and extracted_count < target_frames:
            # Resize frame
            resized_frame = cv2.resize(frame, resize_to)
            
            # Nama file dengan zero padding
            filename = f"frame_{extracted_count:04d}.png"
            filepath = os.path.join(output_folder, filename)
            
            # Simpan sebagai PNG
            cv2.imwrite(filepath, resized_frame)
            extracted_count += 1
            
            if extracted_count % 20 == 0:
                print(f"Extracted {extracted_count} frames...")
        
        frame_number += 1
    
    cap.release()
    print(f"Selesai! Total {extracted_count} frame disimpan di {output_folder}")
    return True

def process_multiple_videos(video_config, base_output_dir="dataset", target_frames=100, resize_to=(224, 224)):
    """
    Proses multiple video sekaligus
    
    Args:
        video_config (dict): Dictionary dengan format {folder_name: video_path}
        base_output_dir (str): Directory utama untuk dataset
        target_frames (int): Jumlah frame per video
        resize_to (tuple): Ukuran resize gambar
    """
    
    # Buat directory utama
    os.makedirs(base_output_dir, exist_ok=True)
    
    print(f"Memproses {len(video_config)} video...")
    print(f"Target frames per video: {target_frames}")
    print(f"Ukuran gambar: {resize_to[0]}x{resize_to[1]}")
    print("-" * 50)
    
    for folder_name, video_path in video_config.items():
        print(f"\nMemproses: {folder_name}")
        output_folder = os.path.join(base_output_dir, folder_name)
        
        if not os.path.exists(video_path):
            print(f"Warning: Video {video_path} tidak ditemukan!")
            continue
            
        success = extract_frames_from_video(video_path, output_folder, target_frames, resize_to)
        
        if success:
            print(f"✅ {folder_name} selesai")
        else:
            print(f"❌ {folder_name} gagal")
    
    print("\n" + "="*50)
    print("SELESAI! Dataset siap untuk training")

# Contoh penggunaan
if __name__ == "__main__":
    # Konfigurasi video Anda - sesuaikan path video
    video_configs = {
        "class_1": "videos/maju.mp4",
        "class_2": "videos/stop.mp4", 
        "class_3": "videos/kiri.mp4",
        "class_4": "videos/kanan.mp4",
    }
    
    # Parameter
    TARGET_FRAMES = 300  # Ubah ke 200 jika ingin 200 frame
    IMAGE_SIZE = (224, 224)  # Ukuran standar untuk training (bisa diubah ke 128x128, 256x256, dll)
    OUTPUT_DIR = "hand_gesture_dataset"
    
    # Jalankan proses
    process_multiple_videos(
        video_config=video_configs,
        base_output_dir=OUTPUT_DIR,
        target_frames=TARGET_FRAMES,
        resize_to=IMAGE_SIZE
    )
    
    # Tampilkan struktur folder yang dihasilkan
    print(f"\nStruktur dataset di '{OUTPUT_DIR}':")
    for folder_name in video_configs.keys():
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        if os.path.exists(folder_path):
            file_count = len([f for f in os.listdir(folder_path) if f.endswith('.png')])
            print(f"  📁 {folder_name}/ - {file_count} files")