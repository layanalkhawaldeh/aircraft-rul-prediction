import os
import sys

# إضافة مجلد src للمسارات
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from preprocessing import load_and_prepare_data

# 1. تحويل البيانات إلى Sequences باستخدام Sliding Window (Part 9)
def create_sequences(df, features, sequence_length=30):
    X_seq, y_seq = [], []
    for engine_id, group in df.groupby('unit_id'):
        feature_data = group[features].values
        rul_data = group['RUL'].values
        
        # إذا كان عدد الدورات للمحرك أقل من طول النافذة، نكتفي بالبيانات المتاحة
        if len(feature_data) < sequence_length:
            continue
            
        for i in range(len(feature_data) - sequence_length + 1):
            X_seq.append(feature_data[i:i + sequence_length])
            y_seq.append(rul_data[i + sequence_length - 1]) # الـ RUL المقابل لآخر دورة بالنافذة
            
    return np.array(X_seq), np.array(y_seq)

# 2. بناء بنية شبكة الـ LSTM (Part 10)
class LSTMRULModel(nn.Module):
                  #عدد الحساسات =16 
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(LSTMRULModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_features)
        lstm_out, (hn, cn) = self.lstm(x)
        # نأخذ المخرجات الخاصة بأخر خطوة زمنية (Last Time Step)
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out

def train_lstm_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    charts_dir = os.path.join(base_dir, "charts", "training_curves")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    # تحميل البيانات الجاهزة
    df, features, _ = load_and_prepare_data(base_dir)

    # تقسيم البيانات حسب المحركات لمنع التسريب (Train: 1-80, Val: 81-100)
    train_df = df[df['unit_id'] <= 80]
    val_df = df[df['unit_id'] > 80]

    # إنشاء السلاسل الزمنية نافذة = 30
    seq_length = 30
    X_train_raw, y_train_raw = create_sequences(train_df, features, sequence_length=seq_length)
    X_val_raw, y_val_raw = create_sequences(val_df, features, sequence_length=seq_length)

    X_train = torch.tensor(X_train_raw, dtype=torch.float32)
    y_train = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val_raw, dtype=torch.float32)
    y_val = torch.tensor(y_val_raw, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    # إعداد الموديل
    model = LSTMRULModel(input_dim=len(features), hidden_dim=64, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_path = os.path.join(models_dir, "best_sequence_model.pt")

    print(f"Training LSTM Sequence Model (Seq Length = {seq_length})...")
    
    epochs = 60
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_loss / len(X_train)

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            epoch_val_loss = criterion(val_preds, y_val).item()

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss (MSE): {epoch_train_loss:.2f} | Val Loss (MSE): {epoch_val_loss:.2f}")

        # Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

    # حفظ منحنى التعلم
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss (MSE)')
    plt.plot(val_losses, label='Val Loss (MSE)')
    plt.title('LSTM Model Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(charts_dir, 'lstm_model_loss.png'))
    plt.close()

    print(f"\nTraining completed! Best LSTM model saved to: {best_model_path}")

if __name__ == "__main__":
    train_lstm_model()