import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def run_eda():
    # 1. Setup paths safely
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "train_FD001.txt")
    
    # Use a dedicated subfolder to prevent file/folder conflicts and keep old charts safe
    charts_dir = os.path.join(base_dir, "charts", "dl_sensor_analysis")
    
    # Safe directory creation without deleting existing files
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir, exist_ok=True)
    
    # 2. Define columns
    col_names = ['unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] 
    + [f'sensor_{i}' for i in range(1, 22)]
    
    # 3. Load data
    df = pd.read_csv(data_path, sep=r'\s+', header=None, names=col_names)
    
    # 4. Calculate Remaining Useful Life (RUL Target - Part 2)
    max_cycles = df.groupby('unit_id')['time_cycles'].transform('max')# بروح على كل محرك وبشوف أقصى دورة وصلها قبل ما يخرب
    df['RUL'] = max_cycles - df['time_cycles']
    
    print("="*50)
    print("EXPLORATORY DATA ANALYSIS (EDA) RESULTS")
    print("="*50)
    
    # How many engines exist?
    num_engines = df['unit_id'].nunique()#بتعطينا عدد المحركات الفريدة بالداتا
    print(f"1. Number of engines: {num_engines}")
    
    # How many cycles does each engine contain? Do they have the same lifetime?
    engine_lifetimes = df.groupby('unit_id')['time_cycles'].max()#بنحسب أقصى عمر لكل محرك عشان نعرف أقل عمر للمحركات وأطول عمر.
    print(f"2. Engine lifetimes (max cycles before failure):")
    print(f"   - Minimum lifetime: {engine_lifetimes.min()} cycles")
    print(f"   - Maximum lifetime: {engine_lifetimes.max()} cycles")
    print(f"   - Average lifetime: {engine_lifetimes.mean():.2f} cycles")
    print(f"   - Do all engines have the same lifetime? {'No' if engine_lifetimes.nunique() > 1 else 'Yes'}")
    
    # Missing values
    missing_vals = df.isnull().sum().sum()#بنفحص إذا في قيم مفقودة أو ممسوحة
    print(f"3. Missing or invalid values: {missing_vals}")
    
    # Identify sensors that are constant vs those that change
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    """
    كيف بنعرف الحساس اللي بشتغل من الحساس الخربان/الثابت؟ عن طريق حساب الانحراف المعياري (Standard Deviation - std).
    constant_sensors (الحساسات الثابتة): قيمتها ثابتة دايماً وما بتتغير (الـ std لها أقل من 0.01)
    ، هدول 7 حساسات ما بستفيد منهم الموديل ورح نحذفهم بالخطوة الجاية عشان ما يشتتوا الذكاء الاصطناعي.
    variable_sensors (الحساسات المتغيرة): قيمتها بتتغير مع الاستخدام وتآكل المحرك (الـ std لها أكبر من 0.01)
    ، هدول 14 حساس هم الـ Features الحقيقيين اللي رح يعلموا الموديل متى المحرك رح يخرب.
    """
    sensor_stds = df[sensor_cols].std()
    constant_sensors = sensor_stds[sensor_stds < 0.01].index.tolist()
    variable_sensors = sensor_stds[sensor_stds >= 0.01].index.tolist()
    
    print("\n4. Sensor Variability Analysis:")
    print(f"   - Constant sensors (std < 0.01): {constant_sensors}")
    print(f"   - Variable sensors (std >= 0.01): {variable_sensors}")
    
    # Save lifetime distribution chart
    plt.figure(figsize=(8, 5))
    plt.hist(engine_lifetimes, bins=15, color='#4A90E2', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Engine Lifetimes (Max Cycles until Failure)')
    plt.xlabel('Cycles to Failure')
    plt.ylabel('Number of Engines')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    #بتفرجينا اعمار ال 100 محرك
    save_path_hist = os.path.join(charts_dir, 'engine_lifetimes_distribution.png')
    plt.savefig(save_path_hist)
    plt.close()
    print(f"\nSaved lifetime distribution chart to {save_path_hist}")
    
    engine_1 = df[df['unit_id'] == 1]#بنفلتر الجدول وبناخد بس بيانات المحرك رقم 1 كعينة.
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)#خلّي كل الرسومات تشترك بنفس المحور السيني
    fig.suptitle('Sensor Readings over Lifetime (Engine 1) - Visualizing Degradation', fontsize=16)
    
    sample_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_12']
    
    for idx, sensor in enumerate(sample_sensors):
        ax = axes[idx // 2, idx % 2]
        ax.plot(engine_1['time_cycles'], engine_1[sensor], color='#E74C3C' if idx % 2 == 0 else '#2ECC71', linewidth=2)
        ax.set_title(f'{sensor} Trajectory')
        ax.set_ylabel('Sensor Value')
        ax.grid(True, linestyle=':', alpha=0.6)
        if idx >= 4:
            ax.set_xlabel('Operating Cycle')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path_trends = os.path.join(charts_dir, 'engine_1_degradation_trends.png')
    plt.savefig(save_path_trends)
    plt.close()
    print(f"Saved Engine 1 degradation trends chart to {save_path_trends}")

    # Save a chart with constant sensors for comparison
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    fig.suptitle('Constant/Flat Sensor Readings over Lifetime (Engine 1)', fontsize=14)
    flat_samples = ['sensor_1', 'sensor_5', 'sensor_16', 'sensor_18']#بنرسم 4 حساسات من الثابتين عشان نثبت بالرسم البياني إنهم عبارة عن خطوط مستقيمة وثابتة وما فيها أي معلومات مفيدة
    
    for idx, sensor in enumerate(flat_samples):
        ax = axes[idx // 2, idx % 2]
        ax.plot(engine_1['time_cycles'], engine_1[sensor], color='#7F8C8D', linewidth=2)
        ax.set_title(f'{sensor} (Constant)')
        ax.set_ylabel('Sensor Value')
        ax.grid(True, linestyle=':', alpha=0.6)
        if idx >= 2:
            ax.set_xlabel('Operating Cycle')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path_flat = os.path.join(charts_dir, 'engine_1_constant_sensors.png')
    plt.savefig(save_path_flat)
    plt.close()
    print(f"Saved Engine 1 constant sensors chart to {save_path_flat}")
    print("="*50)

if __name__ == "__main__":
    run_eda()