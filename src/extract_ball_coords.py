import os
import sys
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import RAW_VIDEO_DIR, FEATURES_CSV_DIR, TRACKNET_DIR, TRACKNET_WEIGHTS, INPAINTNET_WEIGHTS

def main():
    """Hàm chính để chạy TrackNetV3 trên tất cả các video."""
    if not os.path.isdir(TRACKNET_DIR):
        print(f"Lỗi: Không tìm thấy thư mục TrackNetV3 tại '{TRACKNET_DIR}'")
        print("Vui lòng clone repository TrackNetV3 vào thư mục gốc của dự án.")
        return

    print(f"Bắt đầu trích xuất tọa độ cầu từ '{RAW_VIDEO_DIR}' vào '{FEATURES_CSV_DIR}'")
    
    current_working_dir = os.getcwd()
    os.chdir(TRACKNET_DIR) # Phải chuyển vào thư mục TrackNetV3 để chạy predict.py

    for class_folder in os.listdir(RAW_VIDEO_DIR):
        class_path = os.path.join(current_working_dir, RAW_VIDEO_DIR, class_folder)
        if not os.path.isdir(class_path): continue

        output_class_path = os.path.join(current_working_dir, FEATURES_CSV_DIR, class_folder)
        os.makedirs(output_class_path, exist_ok=True)
        print(f"\nĐang xử lý lớp: {class_folder}")

        video_files = [f for f in os.listdir(class_path) if f.endswith('.mp4')]
        for video_file in video_files:
            video_path = os.path.join(class_path, video_file)
            file_name_without_ext = os.path.splitext(video_file)[0]
            
            # TrackNetV3 lưu kết quả với hậu tố _ball.csv
            expected_output_path = os.path.join(output_class_path, f"{file_name_without_ext}_ball.csv")

            if os.path.exists(expected_output_path):
                print(f"  Đã có file tọa độ cầu cho '{video_file}'. Bỏ qua.")
                continue

            print(f"  Đang xử lý video: {video_file}")
            # Xây dựng command để thực thi
            command = [
                "python", "predict.py",
                "--video_file", video_path,
                "--tracknet_file", TRACKNET_WEIGHTS,
                "--inpaintnet_file", INPAINTNET_WEIGHTS,
                "--save_dir", output_class_path
            ]

            # Thực thi command
            # Sử dụng subprocess để có thể xem output trực tiếp
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in iter(process.stdout.readline, ''):
                print(f"    [TrackNetV3]: {line.strip()}")
            process.stdout.close()
            process.wait()
            
            # Đổi tên file output của TrackNet thành định dạng mong muốn
            original_output = os.path.join(output_class_path, f"{file_name_without_ext}.csv")
            if os.path.exists(original_output):
                 os.rename(original_output, expected_output_path)
                 print(f"  -> Đã lưu và đổi tên file tọa độ cầu thành {os.path.basename(expected_output_path)}")


    os.chdir(current_working_dir) # Quay trở lại thư mục làm việc ban đầu
    print("\nHoàn tất trích xuất tọa độ cầu.")

if __name__ == "__main__":
    main()