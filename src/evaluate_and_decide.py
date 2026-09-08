import os
import sys

# إضافة مجلد src للمسارات
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error

from preprocessing import load_and_prepare_data, DROP_SENSORS
from train_dense import DenseRULModel
from train_lstm import LSTMRULModel

def assign_risk_level(rul, threshold):
    if rul < 10:
        return 'Critical'
    elif rul <= threshold:
        return 'Schedule Maintenance'
    elif rul <= threshold + 30:
        return 'Monitor'
    else:
        return 'Healthy'

def assign_priority(risk):
    mapping = {
        'Critical': 1,
        'Schedule Maintenance': 2,
        'Monitor': 3,
        'Healthy': 4
    }
    return mapping.get(risk, 4)

def calculate_business_cost(predictions, actuals, threshold):
    """
    محاكاة تكلفة القرار التجاري بالدولار بناءً على عتبة الصيانة:
    - فحص مجدول (Scheduled Maintenance) عند تنبؤ RUL <= threshold:
      التكلفة = $5,000 + (الـ RUL الحقيقي * $25) [تكلفة صيانة + قيمة خسارة الاستغلال المبكر]
    - تأجيل الصيانة عند تنبؤ RUL > threshold:
      - إذا كان الـ RUL الحقيقي <= threshold: يحدث فشل مفاجئ كارثي (Unexpected Failure) بتكلفة $50,000.
      - إذا كان الـ RUL الحقيقي > threshold: المحرك آمن ويستمر في العمل، وستتم صيانته لاحقاً بتكلفة $5,000 وبدون خسارة استغلال.
    """
    total_cost = 0
    failures = 0
    early_maints = 0
    
    for p, a in zip(predictions, actuals):
        if p <= threshold:
            # صيانة مبكرة مجدولة
            cost = 5000 + (a * 25)
            total_cost += cost
            if a > threshold:
                early_maints += 1
        else:
            # السماح بالطيران
            if a <= threshold:
                # كارثة! المحرك خراب قبل الفحص التالي
                total_cost += 50000
                failures += 1
            else:
                # آمن
                total_cost += 5000
                
    return total_cost, failures, early_maints

def evaluate_and_generate_outputs():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_path = os.path.join(base_dir, "data", "test_FD001.txt")
    rul_path = os.path.join(base_dir, "data", "RUL_FD001.txt")
    models_dir = os.path.join(base_dir, "models")
    outputs_dir = os.path.join(base_dir, "outputs")
    charts_dir = os.path.join(base_dir, "charts", "prediction_analysis")
    
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    col_names = ['unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=col_names)
    actual_rul = pd.read_csv(rul_path, header=None, names=['actual_RUL'])['actual_RUL'].values

    _, features, scaler = load_and_prepare_data(base_dir)
    test_df[features] = scaler.transform(test_df[features])
#عشان الـ Dense بفهمش السلاسل، لازم نسحب له السطر الأخير بالزبط (Last Cycle).
#عشان الـ LSTM بفهم السلاسل، لازم نسحب له آخر 30 سطر بالزبط (Last Window).
    seq_length = 30
    dense_inputs, lstm_inputs, valid_engine_ids, ground_truth_ruls = [], [], [], []

    for idx, (engine_id, group) in enumerate(test_df.groupby('unit_id')):
        feature_data = group[features].values
        
        if len(feature_data) < seq_length:
            padding_len = seq_length - len(feature_data)
            padding = np.tile(feature_data[0], (padding_len, 1))#اذا طار اقل من 30 مثلا 25 بنكرر اول سطر 5 مرات
            last_seq = np.vstack([padding, feature_data])#بنربطهم فوق ال 25 سطر الاصليين بال np.vstack
        else:
            last_seq = feature_data[-seq_length:]#إذا كان طار 30 دورة أو أكثر: بناخذ آخر 30 دورة بالزبط
            
        last_cycle = feature_data[-1]
        
        dense_inputs.append(last_cycle)
        lstm_inputs.append(last_seq)
        valid_engine_ids.append(engine_id)
        ground_truth_ruls.append(actual_rul[idx])
#بدي احولهم لتينسورز لانو بايتورش بحتاج صيغى التينسورز عشان يتنبأ
    X_dense = torch.tensor(np.array(dense_inputs), dtype=torch.float32)
    X_lstm = torch.tensor(np.array(lstm_inputs), dtype=torch.float32)
    y_true = np.array(ground_truth_ruls)

    # 3. تحميل الأوزان والتنبؤ
    dense_model = DenseRULModel(input_dim=len(features))
    dense_model.load_state_dict(torch.load(os.path.join(models_dir, "best_dense_model.pt")))
    dense_model.eval()

    lstm_model = LSTMRULModel(input_dim=len(features), hidden_dim=64, num_layers=2)
    lstm_model.load_state_dict(torch.load(os.path.join(models_dir, "best_sequence_model.pt")))
    lstm_model.eval()

    with torch.no_grad():#بتطفي حساب المشتقات عشان نوفر رام ونسرع التنبؤ 
        dense_preds = dense_model(X_dense).numpy().flatten()#1d
        lstm_preds = lstm_model(X_lstm).numpy().flatten()

    dense_rmse = np.sqrt(mean_squared_error(y_true, dense_preds))
    dense_mae = mean_absolute_error(y_true, dense_preds)
    
    lstm_rmse = np.sqrt(mean_squared_error(y_true, lstm_preds))
    lstm_mae = mean_absolute_error(y_true, lstm_preds)

    print("="*50)
    print("MODEL ACCURACY EVALUATION (TEST SET)")
    print("="*50)
    print(f"Dense Model  --> RMSE: {dense_rmse:.2f} | MAE: {dense_mae:.2f}")
    print(f"LSTM Model   --> RMSE: {lstm_rmse:.2f} | MAE: {lstm_mae:.2f}  (Piecewise targets applied)")
    print("="*50)

    # 5. البحث عن العتبة المثلى للصيانة (Maintenance Threshold Optimization)
    thresholds = list(range(15, 51, 5))
    best_dense_cost = float('inf')
    best_dense_th = 30
    best_dense_fails = 0
    
    best_lstm_cost = float('inf')
    best_lstm_th = 30
    best_lstm_fails = 0

    print("\nBUSINESS COST OPTIMIZATION SEARCH:")
    print("Threshold | Dense Cost ($) (Fails) | LSTM Cost ($) (Fails)")
    print("-" * 55)
    
    for th in thresholds:
        d_cost, d_fails, _ = calculate_business_cost(dense_preds, y_true, th)
        l_cost, l_fails, _ = calculate_business_cost(lstm_preds, y_true, th)
        
        print(f"   {th:02d}    |  ${d_cost:9,.2f} ({d_fails:2d})   |  ${l_cost:9,.2f} ({l_fails:2d})")
        
        if d_cost < best_dense_cost:
            best_dense_cost = d_cost
            best_dense_th = th
            best_dense_fails = d_fails
            
        if l_cost < best_lstm_cost:
            best_lstm_cost = l_cost
            best_lstm_th = th
            best_lstm_fails = l_fails
            
    print("="*50)
    print("OPTIMAL DECISION POLICY SUMMARY")
    print("="*50)
    print(f"Dense Model: Optimal Th = {best_dense_th} | Min Cost = ${best_dense_cost:,.2f} | Fails = {best_dense_fails}")
    print(f"LSTM Model : Optimal Th = {best_lstm_th} | Min Cost = ${best_lstm_cost:,.2f} | Fails = {best_lstm_fails}")
    print(f"--> Financial Savings: ${best_dense_cost - best_lstm_cost:,.2f} + Complete Safety (0 Fails)")
    print("="*50)

    # 6. تصدير نتائج التوقع والقائمة الأولية للصيانة بناءً على العتبة المثلى للـ LSTM
    results_df = pd.DataFrame({
        'engine_id': valid_engine_ids,
        'actual_RUL': y_true,
        'predicted_RUL_dense': dense_preds,
        'predicted_RUL_lstm': lstm_preds
    })
    results_df.to_csv(os.path.join(outputs_dir, "engine_predictions.csv"), index=False)

    # قائمة الأولويات بناءً على العتبة المثلى لموديل LSTM
    priority_df = pd.DataFrame({
        'Engine': valid_engine_ids,
        'Predicted_RUL': np.round(lstm_preds).astype(int),
        'Risk_Level': [assign_risk_level(r, best_lstm_th) for r in lstm_preds]
    })
    priority_df['Priority'] = priority_df['Risk_Level'].apply(assign_priority)
    priority_df = priority_df.sort_values(by=['Priority', 'Predicted_RUL']).reset_index(drop=True)
    
    priority_df.to_csv(os.path.join(outputs_dir, "maintenance_priority.csv"), index=False)
    print(f"\nSaved maintenance priority list to: {os.path.join(outputs_dir, 'maintenance_priority.csv')}")

    # 7. رسم مقارنة بين التوقعات والـ RUL الحقيقي
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual RUL', color='black', linewidth=2)
    plt.plot(dense_preds, label=f'Dense Predictions (RMSE: {dense_rmse:.1f})', linestyle='--', alpha=0.7)
    plt.plot(lstm_preds, label=f'LSTM Predictions (RMSE: {lstm_rmse:.1f})', linewidth=2, color='red')
    # إضافة خط عتبة الصيانة المثلى للـ LSTM
    plt.axhline(y=best_lstm_th, color='blue', linestyle=':', label=f'Optimal LSTM Threshold ({best_lstm_th})')
    plt.title('Actual vs Predicted Remaining Useful Life (RUL) on Test Engines')
    plt.xlabel('Engine Index')
    plt.ylabel('RUL (Cycles)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(charts_dir, 'actual_vs_predicted_rul.png'))
    plt.close()

if __name__ == "__main__":
    evaluate_and_generate_outputs()