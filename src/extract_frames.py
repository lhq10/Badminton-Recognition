import os
import cv2
import sys

# Thêm thư mục src vào path để import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import RAW_VIDEO_DIR, PROCESSED_FRAMES_DIR

def extract_all_frames(video_path, output_folder):
    """Trích xuất tất cả các frame từ một video và lưu chúng."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        frame_path = os.path.join(output_folder, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"-> Đã trích xuất {frame_count} frames từ {os.path.basename(video_path)}")

def process_all_videos(input_root, output_root):
    """Xử lý tất cả video trong các thư mục con."""
    os.makedirs(output_root, exist_ok=True)
    print(f"Bắt đầu quá trình trích xuất frame từ '{input_root}' vào '{output_root}'")

    for class_name in os.listdir(input_root):
        class_input = os.path.join(input_root, class_name)
        if not os.path.isdir(class_input):
            continue
            
        class_output = os.path.join(output_root, class_name)
        os.makedirs(class_output, exist_ok=True)
        print(f"\nĐang xử lý lớp: {class_name}")

        for video in os.listdir(class_input):
            if not video.endswith(('.mp4', '.avi', '.mov')):
                continue
            
            name = os.path.splitext(video)[0]
            video_path = os.path.join(class_input, video)
            output_folder = os.path.join(class_output, name)
            os.makedirs(output_folder, exist_ok=True)

            extract_all_frames(video_path, output_folder)
            
    print("\nHoàn tất việc trích xuất frames cho tất cả video.")

if __name__ == "__main__":
    process_all_videos(RAW_VIDEO_DIR, PROCESSED_FRAMES_DIR)