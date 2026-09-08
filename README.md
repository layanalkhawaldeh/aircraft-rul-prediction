# ✈️ Aircraft Engine RUL Prediction

A deep learning-based predictive maintenance system for predicting the **Remaining Useful Life (RUL)** of aircraft engines using the **NASA C-MAPSS FD001 dataset**.

The project compares a baseline Dense Neural Network with an LSTM-based sequence model to estimate engine degradation and support proactive maintenance decisions.

---

## 📌 Overview

Aircraft engine maintenance is a critical predictive analytics problem where accurately estimating how much useful operating life remains can help identify engines that require attention before failure.

This project builds an end-to-end predictive maintenance pipeline that processes multivariate engine sensor data, trains deep learning models, predicts Remaining Useful Life, and generates maintenance-priority outputs.

The system includes:

- Data preprocessing and sensor selection
- Remaining Useful Life target generation
- Leakage-safe feature scaling
- Dense Neural Network baseline
- LSTM-based time-series modeling
- Model evaluation and comparison
- Engine-level RUL prediction
- Maintenance risk prioritization
- PostgreSQL and Docker integration

---

## 🧠 Models

### Dense Neural Network

A feed-forward neural network is used as a baseline model for RUL regression.

Architecture:

- Dense layer — 64 units
- Dense layer — 32 units
- Dropout — 0.2
- Regression output layer

### LSTM Sequence Model

The sequence model uses **30-cycle sliding windows** to capture temporal degradation patterns in aircraft engine sensor readings.

Architecture:

- Two-layer LSTM
- Hidden size — 64
- Dense layer — 32 units
- Dropout
- Regression output layer

---

## 📊 Model Performance

| Model | RMSE | MAE |
|---|---:|---:|
| Dense Neural Network | 18.14 | 13.33 |
| **LSTM Sequence Model** | **14.76** | **11.43** |

The LSTM model achieved better performance than the baseline Dense Neural Network, demonstrating the value of modeling temporal dependencies in engine degradation data.

---

## 🔧 Predictive Maintenance Simulation

The predicted RUL values are also used in a maintenance simulation to identify engines requiring priority attention.

Using a maintenance threshold of **45 cycles**, the simulation produced:

| Model | Simulated Cost | Failures |
|---|---:|---:|
| Dense Neural Network | $647,125 | 3 |
| LSTM | $514,975 | 0 |

In this project simulation, the LSTM-based strategy resulted in **$132,150 lower simulated cost** while avoiding simulated engine failures.

> **Note:** These values are outputs of the project's maintenance simulation and should not be interpreted as real-world operational or financial guarantees.

---

## 📂 Dataset

The project uses the **NASA C-MAPSS FD001** turbofan engine degradation dataset.

The dataset contains multiple engine trajectories consisting of operational settings and sensor measurements recorded over successive operating cycles.

For FD001:

- 100 engine trajectories
- 21 sensor measurements
- 3 operational settings
- Run-to-failure training trajectories
- Test trajectories with corresponding RUL targets

During preprocessing, non-informative sensors are removed and the remaining degradation-related sensor features are used for model training.

Raw dataset files are intentionally excluded from this repository.

---

## ⚙️ Data Processing Pipeline

The main processing workflow includes:

1. Load C-MAPSS engine sensor data
2. Remove non-informative sensor features
3. Generate Remaining Useful Life targets
4. Cap RUL values at 125 cycles
5. Split engines into training and validation sets
6. Apply leakage-safe MinMax scaling
7. Generate 30-cycle sequences for LSTM training
8. Train and evaluate deep learning models
9. Generate engine-level predictions
10. Produce maintenance-priority outputs

---

## 🛠️ Tech Stack

**Programming & Machine Learning**

- Python
- PyTorch
- NumPy
- Pandas
- scikit-learn

**Data & Infrastructure**

- PostgreSQL
- Docker
- Docker Compose

**Modeling**

- Deep Learning
- LSTM Networks
- Time-Series Regression
- Predictive Maintenance
- Remaining Useful Life Prediction

---

## 📁 Project Structure

```text
aircraft-rul-prediction/
│
├── charts/                 # Generated model visualizations
├── models/                 # Trained model weights
├── outputs/                # Prediction and maintenance outputs
├── reports/                # Project reports and evaluation results
├── src/                    # Source code
│
├── .env.example            # Example environment configuration
├── .gitignore
├── docker-compose.yml      # PostgreSQL container configuration
└── README.md
```

Raw dataset files and the local database are excluded from version control.

---

## 🗄️ Database & Docker

PostgreSQL is configured through Docker Compose.

Create a local `.env` file using `.env.example`:

```env
POSTGRES_USER=your_postgres_user
POSTGRES_PASSWORD=your_postgres_password
POSTGRES_DB=your_database_name
```

Then start PostgreSQL with:

```bash
docker compose up -d
```

Sensitive local credentials are not stored in the repository.

---

## 📈 Outputs

The pipeline generates engine-level predictive maintenance outputs including:

```text
outputs/
├── engine_predictions.csv
└── maintenance_priority.csv
```

These outputs can be used to inspect predicted Remaining Useful Life and prioritize engines according to maintenance risk.

---

## 🔬 Key Concepts Demonstrated

This project demonstrates practical experience with:

- Deep learning for regression
- Sequential time-series modeling
- LSTM neural networks
- Predictive maintenance
- Feature preprocessing
- Data leakage prevention
- Model evaluation using RMSE and MAE
- Maintenance risk simulation
- Model persistence
- Database integration
- Containerized infrastructure

---

## 🚀 Future Improvements

Potential improvements include:

- Experimenting with GRU and Transformer-based sequence models
- Hyperparameter optimization
- Additional C-MAPSS subsets such as FD002–FD004
- Uncertainty estimation for RUL predictions
- Real-time sensor ingestion
- Interactive maintenance dashboard
- Model serving through a REST API
- Experiment tracking and production monitoring

---

## 👩‍💻 Author

**Layan Alkhawaldeh**  
AI Engineer | Artificial Intelligence & Data Science

GitHub: [layanalkhawaldeh](https://github.com/layanalkhawaldeh)

LinkedIn: [Layan Alkhawaldeh](https://www.linkedin.com/in/layan-alkhawaldeh-025977325/)
