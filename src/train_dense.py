import os
import sys

# إضافة مجلد src للمسارات لضمان التعرف على preprocessing.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from preprocessing import load_and_prepare_data

# 1. بناء شبكة الـ Dense باستخدام PyTorch (Part 4)
#Feed-Forward Dense Network)
class DenseRULModel(nn.Module):
    def __init__(self, input_dim):
        super(DenseRULModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),#بتاخد 16 سينسور وبتضرب كل واحد بوزن وبايس وبتطلعلنا 64 قيمة جديدة 
            nn.ReLU(),#اكتيفيشن فنكشن بتخلي السالب صفر والموجب موجب
            nn.Dropout(0.2),#طبقة حماية بتطفي 20 بالميى من النيورونز عشوائي بتخليهم صفر عشان تمنع الاوفر فتنغ
            nn.Linear(64, 32),#لاير تانية بتاخد ال 64 قيمة السابقة وبتصغرهم ا 32 قيمة
            #بتاخد اهم المعلومات والميزات 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1) # Output RUL
        )#بتاخد ال 32 قيمة وبتجمعهم وبتضربهم باوزان عشان تطلعلنا قيمة وحدة بس 

    def forward(self, x):
        return self.net(x)

def train_dense_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    charts_dir = os.path.join(base_dir, "charts", "training_curves")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    # 2. تحميل البيانات وتجهيزها
    df, features, scaler = load_and_prepare_data(base_dir)

    # تقسيم البيانات حسب المحركات لتجنب تسريب البيانات (Part 6)
    train_df = df[df['unit_id'] <= 80]
    val_df = df[df['unit_id'] > 80]

    X_train = torch.tensor(train_df[features].values, dtype=torch.float32)
    y_train = torch.tensor(train_df['RUL'].values, dtype=torch.float32).unsqueeze(1)
    
    X_val = torch.tensor(val_df[features].values, dtype=torch.float32)
    y_val = torch.tensor(val_df['RUL'].values, dtype=torch.float32).unsqueeze(1)
                                                              # بيدرس باتشيز (دفعات) يعني مش 16 الف مرة وحدة ولا وحدة وحدة
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)#نخلط كلشي عشان نجبرو يفهم ما يحفظ بالترتيب

    model = DenseRULModel(input_dim=len(features))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)#بيدرس ال LOss وبعدل الخلايا والاوزان
    #lr=0.001 (Learning Rate): هي سرعة خطوة التعلم للوزن. إذا كانت كبيرة الموديل بخرب، وإذا صغيرة بصير بطيء جداً.
    #  قيمة 0.001 هي الرقم المثالي المجرب عالمياً لمحسن Adam.

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_path = os.path.join(models_dir, "best_dense_model.pt")

    print("Training Dense Neural Network (PyTorch)...")
    
    # 4. تدريب الموديل (Part 7 & Part 8)
    epochs = 60 #مسموح للموديل يلف على كل الباتشز 60 مرة كحد اقصى
    patience = 10#اذا الموديل تعلم ومرت 10 ايبوكس وما تحسن الفاليديشن وقف التدريب لانو رح يصير يحفظ حفظ
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            #batch_x: قراءات الحساسات الـ 32 (الأسئلة).
            #batch_y: الـ RUL الحقيقي الـ 32 (الأجوبة الصح).
            optimizer.zero_grad()#لازم نصفر ال gradients عشان الموديل ما يجمع الاخطاء الجديدة عالقديمة وتخرب الاوزان
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)# (criterion) بقارن تنبؤات الموديل بالأجوبة الصح، وبحسب الـ MSE (نسبة الخطأ).
            loss.backward()
            # الموديل بيرجع بظهره من المخرجات للمدخلات. وبحسب مشتقة الخطأ بالنسبة لكل وزن (يعني بقيس: "كم كل خلية عصبية ساهمت في ارتكاب هذا الخطأ؟").
            #  هاي العملية السحرية اللي بتخلي الشبكة العصبية تعرف وين غلطت بالزبط.
            optimizer.step()
            # المدرب (optimizer) بستخدم نسب المساهمة في الخطأ اللي انحسبت بالخطوة السابقة، وبعدل الأوزان بخطوة بسيطة عشان المرة الجاية يقل الخطأ.
            running_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_loss / len(X_train)

        # Validation phase
        model.eval()#بوقف الـ Dropout تماماً وبيجعل كل الخلايا تشتغل 100% (لأننا بالامتحان وبدنا أحسن إجابة بدون تشويش).
        with torch.no_grad():
            val_preds = model(X_val)#بنعطي الموديل اسئلة الفاليديشن وبخليه يتوقع الاجوبز
            epoch_val_loss = criterion(val_preds, y_val).item()#بقيم الاجوبة بحسب معدل الخطا فيها

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss (MSE): {epoch_train_loss:.2f} | Val Loss (MSE): {epoch_val_loss:.2f}")

        # Early Stopping & Checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0 #بنصفر العداد لانه الطالب اثبت تحسنه بنعطيه فرصة تانية 
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

    # 5. حفظ منحنى التعلم (Training Curves)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss (MSE)')
    plt.plot(val_losses, label='Val Loss (MSE)')
    plt.title('Dense Model Training & Validation Loss (PyTorch)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(charts_dir, 'dense_model_loss.png'))
    plt.close()

    print(f"\nTraining completed! Best Dense model saved to: {best_model_path}")

if __name__ == "__main__":
    train_dense_model()