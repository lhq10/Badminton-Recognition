import torch

# --- 1. ĐƯỜNG DẪN THƯ MỤC ---
# Đường dẫn bạn cần thay đổi cho phù hợp với máy tính của mình
BASE_DIR = ".." # Thư mục gốc của dự án (Badminton_Action_Recognition)

# Đường dẫn dữ liệu đầu vào và đầu ra
RAW_VIDEO_DIR = f"{BASE_DIR}/data/raw_videos"
PROCESSED_FRAMES_DIR = f"{BASE_DIR}/data/processed_frames"
FEATURES_CSV_DIR = f"{BASE_DIR}/data/features_csv"

# --- 2. THAM SỐ TRÍCH XUẤT ĐẶC TRƯNG ---
# Đường dẫn tới thư mục TrackNetV3 đã clone
TRACKNET_DIR = f"{BASE_DIR}/TrackNetV3"
TRACKNET_WEIGHTS = f"{TRACKNET_DIR}/TrackNet_best.pt"
INPAINTNET_WEIGHTS = f"{TRACKNET_DIR}/InpaintNet_best.pt"


# --- 3. THAM SỐ HUẤN LUYỆN ---
# Tham số mô hình
NUM_CLASSES = 9
# 24 (tọa độ tương đối) + 24 (chuyển động keypoints) + 2 (tọa độ cầu) + 2 (chuyển động cầu)
COMBINED_INPUT_DIM = 52

# Tham số huấn luyện
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 5e-4
EPOCHS = 200
PATIENCE = 20 # Số epochs để chờ trước khi dừng sớm

# Thiết bị
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tên file model tốt nhất sẽ được lưu
BEST_MODEL_NAME = "badminton_action_classifier.pth"