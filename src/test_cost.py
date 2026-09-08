import pandas as pd
import numpy as np
df = pd.read_csv("D:/student_intelligence_system/Deep-Learning/outputs/engine_predictions.csv")
def calculate_cost(pred_rul, actual_rul):
    total_cost = 0
    failures = 0
    early_maints = 0
    ontime_maints = 0
    
    for p, a in zip(pred_rul, actual_rul):
        if p <= 30: # Model schedules maintenance
            cost = 5000 + (a * 25) # Maintenance cost + wasted life
            total_cost += cost
            if a <= 30:
                ontime_maints += 1
            else:
                early_maints += 1
        else: # Model says keep flying
            if a <= 10: # Critically close to failure, so it fails unexpectedly
                total_cost += 50000
                failures += 1
            else: # Safe for now, will do maintenance later at optimal time
                total_cost += 5000 
                ontime_maints += 1
                
    return total_cost, failures, early_maints
dense_cost, dense_fail, dense_early = calculate_cost(df['predicted_RUL_dense'], df['actual_RUL'])
lstm_cost, lstm_fail, lstm_early = calculate_cost(df['predicted_RUL_lstm'], df['actual_RUL'])
# Oracle cost (perfect prediction at RUL = 10)
oracle_cost = 100 * (5000 + 10 * 25)
print(f"Dense Model: Total Cost = ${dense_cost:,.2f} | Failures = {dense_fail} | Early Maintenance = {dense_early}")
print(f"LSTM Model: Total Cost = ${lstm_cost:,.2f} | Failures = {lstm_fail} | Early Maintenance = {lstm_early}")
print(f"Oracle (Perfect): Total Cost = ${oracle_cost:,.2f}")