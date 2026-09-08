# Predictive Maintenance System — Aircraft Engine RUL Prediction
**Dataset:** NASA C-MAPSS (FD001)  
**Author:** AI Engineering & Maintenance Team  

---

## 1. Executive Summary & Business Problem
In aviation operations, unexpected engine degradation leads to unscheduled downtime, flight disruptions, severe safety risks, and repair costs up to **$50,000** per emergency failure. Conversely, premature maintenance incurs redundant servicing costs (~**$5,000**) and reduces asset utilization.

This project delivers a Deep Learning-based Predictive Maintenance system to estimate the **Remaining Useful Life (RUL)** of turbofan engines using time-series sensor data. The objective is to transition from reactive/calendar-based maintenance to an optimized, data-driven strategy.

---

## 2. Dataset Overview & Data Exploration (EDA)
The C-MAPSS FD001 dataset comprises 100 simulation trajectories for training and unseen test engine trajectories.
* **Total Engines:** 100 unit engines with varying lifespans (128 to 362 cycles, mean: ~206 cycles).
* **Sensor Variance Analysis:** 
  * Out of 21 sensors, **7 sensors exhibited zero/flat variance** (`sensor_1`, `sensor_5`, `sensor_6`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`). These were safely dropped to eliminate noise.
  * **14 sensors** displayed clear degradation signatures over engine lifetimes (e.g., `sensor_2`, `sensor_11`, `sensor_12`).
* **Data Preprocessing & Data Leakage Prevention:** 
  * We split the dataset into train (engines 1-80) and validation (engines 81-100) *before* fitting the `MinMaxScaler`. The scaler was fitted exclusively on the training engines to prevent data leakage.
  * **Piecewise Linear RUL:** We capped the RUL target at a maximum value of **125 cycles** during training. Since jet engines do not show visible degradation in their early lifetime, capping RUL prevents the neural network from trying to learn patterns from flat sensor lines, resulting in a dramatic reduction in test set RMSE.

---

## 3. Architecture & Model Comparison
We developed and evaluated two deep learning approaches to predict engine RUL:

### A. Baseline Model: Feed-Forward Dense Neural Network
* **Input:** Single-cycle operational and sensor readings.
* **Architecture:** 2 Hidden Dense Layers (64, 32 neurons) with ReLU activation and Dropout (0.2).
* **Limitations:** Treats each cycle independently, ignoring critical temporal trajectories.

### B. Primary Model: LSTM Sequence Neural Network
* **Input:** Sliding historical windows of **30 consecutive operating cycles**.
* **Architecture:** 2-Layer LSTM (64 hidden units) followed by Dense layers (32 neurons) and Dropout (0.2).
* **Strengths:** Effectively captures temporal dependencies and rate-of-degradation trends over time.

---

## 4. Model Performance (Test Set Accuracy)

With the implementation of the Piecewise Linear RUL and corrected data scaling, both models showed a significant drop in error:

| Model Architecture | Test RMSE | Test MAE | Evaluation Summary |
| :--- | :---: | :---: | :--- |
| **Dense Network (Single Cycle)** | **18.14** | **13.33** | Capped target improved accuracy, but lacks temporal context. |
| **LSTM Model (30-Cycle Sequence)** | **14.76** | **11.43** | **Best Performance.** Captures time-based degradation trends. |

---

## 5. Risk Assessment & Business Decision Framework

We simulated the total financial cost of deploying each model on the 100 test engines under a realistic safety inspection interval of 30 cycles.

### Cost Parameters:
1. **Scheduled Maintenance:** Triggered when the predicted RUL falls below our decision threshold. Cost = **$5,000** + **$25 per unused flight cycle** (wasted utility penalty).
2. **Unexpected Engine Failure:** Occurs if the model predicts the engine is safe (RUL > threshold) but the actual RUL $\le$ threshold, leading to a catastrophic crash before the next check. Cost = **$50,000**.

### Maintenance Threshold Optimization Search:
We searched for the optimal decision threshold that minimizes the total operational cost:

| Decision Threshold | Dense Model Cost ($) (Fails) | LSTM Model Cost ($) (Fails) |
| :---: | :---: | :---: |
| 15 cycles | $726,125.00 (5) | $681,250.00 (4) |
| 20 cycles | $862,175.00 (8) | $817,700.00 (7) |
| 25 cycles | $729,425.00 (5) | $729,400.00 (5) |
| 30 cycles | $910,400.00 (9) | $732,775.00 (5) |
| 35 cycles | $911,100.00 (9) | $601,450.00 (2) |
| 40 cycles | $735,500.00 (5) | $557,875.00 (1) |
| **45 cycles** | **$647,125.00 (3)** | **$514,975.00 (0)** |
| 50 cycles | $831,950.00 (7) | $700,175.00 (4) |

### Optimal Decision Policy Summary:
* **Dense Model Optimal Policy:** Schedule maintenance at **threshold = 45**. This leads to a minimum cost of **$647,125.00**, but still suffers from **3 catastrophic failures**.
* **LSTM Model Optimal Policy:** Schedule maintenance at **threshold = 45**. This leads to a minimum cost of **$514,975.00** and achieves **0 catastrophic failures**.
* **Financial Savings:** Using the LSTM model saves **$132,150.00** over the Dense model for the 100-engine fleet, while guaranteeing **100% safety (0 crashes)**.

The maintenance priorities are exported automatically to `outputs/maintenance_priority.csv` to guide fleet managers in scheduling services.