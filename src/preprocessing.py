import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# المستشعرات الثابتة التي حددناها في EDA لاستبعادها
DROP_SENSORS = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']

def load_and_prepare_data(base_dir, piecewise_limit=125):
    """
    تحميل وتجهيز البيانات مع تجنب تسريب البيانات (Data Leakage)
    عبر تطبيق التحجيم (Scaling) بناءً على بيانات التدريب فقط (unit_id <= 80).
    مع إضافة خيار Piecewise Linear RUL لتحسين التنبؤ.
    """
    train_path = os.path.join(base_dir, "data", "train_FD001.txt")
    
    col_names = ['unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    df = pd.read_csv(train_path, sep=r'\s+', header=None, names=col_names)
    
    # 1. Calculate Target RUL
    max_cycles = df.groupby('unit_id')['time_cycles'].transform('max')
    df['RUL_linear'] = max_cycles - df['time_cycles']
    
    if piecewise_limit is not None:
        # Piecewise RUL: تثبيت القيمة عند حد أقصى لأن المحرك لا يبدأ بالتدهور فوراً
        df['RUL'] = df['RUL_linear'].clip(upper=piecewise_limit)
    else:
        df['RUL'] = df['RUL_linear']
    #هاد العمود يلي قبل ما نثبته ع 125
    df = df.drop(columns=['RUL_linear'])
    
    # 2. Drop constant sensors
    features = [c for c in df.columns if c not in ['unit_id', 'time_cycles', 'RUL'] + DROP_SENSORS]
    
    # 3. Fit Scaler ONLY on train split (unit_id <= 80) to prevent validation leakage
    scaler = MinMaxScaler()
    #بنعمل قناع (Mask) اسمه train_indices بكون True فقط للأسطر التابعة لأول 80 محرك (بيانات التدريب)، 
    # و False لمحركات التحقق (من 81 لـ 100).
    train_indices = df['unit_id'] <= 80
    
    # Fit on train features
    #الفيت يعني احسب المن والماكس لكل عمود
    #انا بحكي للسكيلر احسبهم بس من التدريب 
    scaler.fit(df.loc[train_indices, features])
    
    # Transform whole dataframe
    df[features] = scaler.transform(df[features])
    
    return df, features, scaler

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df, features, _ = load_and_prepare_data(base_dir)
    print(f"Data preprocessed successfully! Remaining features: {len(features)}")
    print(f"Sample RUL target (Piecewise capped at 125): {df['RUL'].head().tolist()}")







    """
    ثخةئعلا9اغهصضئةى 
    انا احببببببببببببك 
    """