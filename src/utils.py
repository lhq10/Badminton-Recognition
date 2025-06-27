import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt
import seaborn as sns


def load_and_process_features(keypoints_path, ball_path):
    """Tải và xử lý features từ file keypoints và ball."""
    try:
        keypoints_df = pd.read_csv(keypoints_path)
        try:
            ball_df = pd.read_csv(ball_path)
            if ball_df.empty or 'X' not in ball_df.columns or 'Y' not in ball_df.columns:
                num_frames = len(keypoints_df)
                ball_df = pd.DataFrame(0, index=np.arange(num_frames), columns=['X', 'Y'])
        except (pd.errors.EmptyDataError, FileNotFoundError):
            num_frames = len(keypoints_df)
            ball_df = pd.DataFrame(0, index=np.arange(num_frames), columns=['X', 'Y'])

        keypoint_cols = [c for c in keypoints_df.columns if c.endswith('_x') or c.endswith('_y')]
        raw_keypoints = keypoints_df[keypoint_cols]
        raw_ball = ball_df[['X', 'Y']]

        # 1. Đặc trưng HÌNH DÁNG (Tọa độ tương đối)
        hip_center_x = (raw_keypoints['L_hip_x'] + raw_keypoints['R_hip_x']) / 2
        hip_center_y = (raw_keypoints['L_hip_y'] + raw_keypoints['R_hip_y']) / 2
        relative_keypoints = pd.DataFrame()
        for col in keypoint_cols:
            if '_x' in col: relative_keypoints[col] = raw_keypoints[col] - hip_center_x
            else: relative_keypoints[col] = raw_keypoints[col] - hip_center_y

        # 2. Đặc trưng CHUYỂN ĐỘNG (Vận tốc)
        keypoint_velocities = raw_keypoints.diff().fillna(0)
        ball_velocities = raw_ball.diff().fillna(0)

        # 3. Kết hợp tất cả đặc trưng
        min_len = min(len(relative_keypoints), len(keypoint_velocities), len(raw_ball), len(ball_velocities))
        combined_features = np.concatenate([
            relative_keypoints.iloc[:min_len].values,
            keypoint_velocities.iloc[:min_len].values,
            raw_ball.iloc[:min_len].values,
            ball_velocities.iloc[:min_len].values
        ], axis=1)
        return torch.FloatTensor(combined_features)
    except Exception as e:
        print(f"Lỗi khi xử lý file: {keypoints_path}. Bỏ qua. Lỗi: {e}")
        return None

class BadmintonActionDataset(Dataset):
    def __init__(self, data_dir):
        self.samples, self.labels = [], []
        self.label_encoder = LabelEncoder()
        class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.label_encoder.fit(class_names)
        self.label_map = {i: cls for i, cls in enumerate(self.label_encoder.classes_)}

        print("Đang tải dữ liệu...")
        for class_name in class_names:
            class_dir = os.path.join(data_dir, class_name)
            label = self.label_encoder.transform([class_name])[0]
            keypoints_files = glob.glob(os.path.join(class_dir, "*_keypoints.csv"))
            for kp_file in keypoints_files:
                base_name = os.path.basename(kp_file).replace("_keypoints.csv", "")
                ball_file = os.path.join(class_dir, f"{base_name}_ball.csv")
                if os.path.exists(ball_file):
                    self.samples.append((kp_file, ball_file))
                    self.labels.append(label)
        print(f"Đã tìm thấy {len(self.samples)} mẫu dữ liệu từ {len(class_names)} lớp.")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        keypoints_path, ball_path = self.samples[idx]
        label = self.labels[idx]
        sequence = load_and_process_features(keypoints_path, ball_path)
        if sequence is None: return None, None, None
        return sequence, torch.tensor(label, dtype=torch.long), sequence.shape[0]

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch: return torch.tensor([]), torch.tensor([]), torch.tensor([])
    sequences, labels, lengths = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return padded_sequences, labels, lengths

def plot_history(history, save_path="."):
    """Vẽ đồ thị loss và accuracy từ history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    ax1.plot(history['train_acc'], label='Train Accuracy', color='blue')
    ax1.plot(history['val_acc'], label='Validation Accuracy', color='orange')
    ax1.set_title('Training and Validation Accuracy', fontsize=16)
    ax1.legend(); ax1.grid(True)
    ax2.plot(history['train_loss'], label='Train Loss', color='blue')
    ax2.plot(history['val_loss'], label='Validation Loss', color='orange')
    ax2.set_title('Training and Validation Loss', fontsize=16)
    ax2.legend(); ax2.grid(True)
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="."):
    """Vẽ và lưu ma trận nhầm lẫn."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title('Confusion Matrix', fontsize=16)
    ax.set_xlabel('Predicted Label'); ax.set_ylabel('True Label')
    plt.xticks(rotation=45); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.show()