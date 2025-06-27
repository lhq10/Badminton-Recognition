import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Thêm thư mục src vào Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import từ các file khác trong dự án
from config import (FEATURES_CSV_DIR, BASE_DIR, NUM_CLASSES, COMBINED_INPUT_DIM,
                    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS, PATIENCE, DEVICE, BEST_MODEL_NAME)
from model import ActionClassifier, EarlyStopping
from utils import BadmintonActionDataset, collate_fn, plot_history, plot_confusion_matrix

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct_predictions, total_samples = 0.0, 0, 0
    for sequences, labels, lengths in dataloader:
        if sequences.nelement() == 0: continue
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences, lengths, NUM_CLASSES)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * sequences.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
    return running_loss / total_samples, correct_predictions / total_samples

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct_predictions, total_samples = 0.0, 0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for sequences, labels, lengths in dataloader:
            if sequences.nelement() == 0: continue
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences, lengths, NUM_CLASSES)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    return running_loss / total_samples, correct_predictions / total_samples, all_labels, all_preds

def main():
    """Hàm chính điều phối toàn bộ quá trình."""
    print(f"Sử dụng thiết bị: {DEVICE}")
    full_dataset = BadmintonActionDataset(FEATURES_CSV_DIR)

    if len(full_dataset) == 0:
        print("Không tìm thấy dữ liệu đã xử lý. Vui lòng chạy các script tiền xử lý trước.")
        return

    # Chia dữ liệu
    indices = list(range(len(full_dataset)))
    labels = full_dataset.labels
    train_indices, temp_indices, _, _ = train_test_split(indices, labels, test_size=0.3, random_state=42, stratify=labels)
    val_indices, test_indices, _, _ = train_test_split(temp_indices, [labels[i] for i in temp_indices], test_size=0.5, random_state=42, stratify=[labels[i] for i in temp_indices])

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    test_subset = Subset(full_dataset, test_indices)
    print(f"Số mẫu Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}")

    # Tạo Dataloader
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Khởi tạo mô hình
    model = ActionClassifier(input_dim=COMBINED_INPUT_DIM, hidden_dim=256, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True)
    
    best_model_path = os.path.join(BASE_DIR, BEST_MODEL_NAME)
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=best_model_path)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Vòng lặp huấn luyện
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        scheduler.step(val_loss)

    # Đánh giá và hiển thị kết quả
    print("\n--- Huấn luyện hoàn tất ---")
    plot_history(history, save_path=BASE_DIR)

    print("\n--- Đánh giá trên tập Test với model tốt nhất ---")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_labels, test_preds = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    class_names = [full_dataset.label_map[i] for i in range(NUM_CLASSES)]
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names, zero_division=0))

    plot_confusion_matrix(test_labels, test_preds, class_names, save_path=BASE_DIR)

if __name__ == "__main__":
    main()