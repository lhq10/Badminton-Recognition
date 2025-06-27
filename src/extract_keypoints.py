import os
import cv2
import numpy as np
from rtmlib import Wholebody
import csv
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import PROCESSED_FRAMES_DIR, FEATURES_CSV_DIR, DEVICE

def get_main_person_keypoints(keypoints_per_frame, muscle_indices):
    """Tìm người chơi chính (mắt cá chân phải thấp nhất) và trích xuất keypoints."""
    highest_r_ankle_y = -1
    person_with_highest_r_ankle_idx = -1
    r_ankle_index = 16

    if keypoints_per_frame is not None and len(keypoints_per_frame) > 0:
        for person_idx, person_keypoints in enumerate(keypoints_per_frame):
            if len(person_keypoints) > r_ankle_index:
                r_ankle_y = person_keypoints[r_ankle_index][1]
                if r_ankle_y > highest_r_ankle_y:
                    highest_r_ankle_y = r_ankle_y
                    person_with_highest_r_ankle_idx = person_idx

    if person_with_highest_r_ankle_idx != -1:
        muscle_kpts = keypoints_per_frame[person_with_highest_r_ankle_idx][muscle_indices]
        return muscle_kpts.flatten().tolist()
    else:
        return [0] * (len(muscle_indices) * 2)

def main():
    """Hàm chính để xử lý trích xuất keypoints."""
    print("Thiết lập mô hình Wholebody...")
    wholebody = Wholebody(to_openpose=False, mode='balanced', backend='onnxruntime', device=DEVICE)
    
    muscle_keypoint_names = {
        5: "L_shoulder", 6: "R_shoulder", 7: "L_elbow", 8: "R_elbow",
        9: "L_wrist", 10: "R_wrist", 11: "L_hip", 12: "R_hip",
        13: "L_knee", 14: "R_knee", 15: "L_ankle", 16: "R_ankle"
    }
    muscle_indices = list(muscle_keypoint_names.keys())

    os.makedirs(FEATURES_CSV_DIR, exist_ok=True)
    print(f"Bắt đầu trích xuất keypoints từ '{PROCESSED_FRAMES_DIR}' vào '{FEATURES_CSV_DIR}'")

    for class_folder in os.listdir(PROCESSED_FRAMES_DIR):
        class_path = os.path.join(PROCESSED_FRAMES_DIR, class_folder)
        if not os.path.isdir(class_path): continue

        print(f"\nĐang xử lý lớp: {class_folder}")
        class_output_dir = os.path.join(FEATURES_CSV_DIR, class_folder)
        os.makedirs(class_output_dir, exist_ok=True)

        for action_folder in os.listdir(class_path):
            action_path = os.path.join(class_path, action_folder)
            if not os.path.isdir(action_path): continue

            output_csv_path = os.path.join(class_output_dir, f"{action_folder}_keypoints.csv")
            if os.path.exists(output_csv_path):
                print(f"  Đã có file keypoints cho '{action_folder}'. Bỏ qua.")
                continue
            
            print(f"  Đang xử lý hành động: {action_folder}")
            frame_files = sorted([f for f in os.listdir(action_path) if f.endswith(('.jpg', '.jpeg', '.png'))])

            with open(output_csv_path, 'w', newline='') as csvfile:
                header = ['frame_file']
                for index in sorted(muscle_keypoint_names.keys()):
                    name = muscle_keypoint_names[index].replace(" ", "_")
                    header.extend([f'{name}_x', f'{name}_y'])
                writer = csv.writer(csvfile)
                writer.writerow(header)

                for frame_file in frame_files:
                    frame_path = os.path.join(action_path, frame_file)
                    img = cv2.imread(frame_path)
                    if img is None: continue

                    keypoints_single_frame, _ = wholebody(img)
                    kpts_data = get_main_person_keypoints(keypoints_single_frame, muscle_indices)
                    writer.writerow([frame_file] + kpts_data)
            
            print(f"  -> Đã lưu keypoints vào {os.path.basename(output_csv_path)}")
    print("\nHoàn tất trích xuất keypoints.")

if __name__ == "__main__":
    main()