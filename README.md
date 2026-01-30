# Chaos-Driven Quantitative Trading on Volatile Assets

## 📌 Overview
This project investigates the use of **chaotic dynamical systems** as feature generators for **machine learning–driven trading strategies** on highly volatile assets (BTC, ETH, SHIB).  
By simulating deterministic yet unpredictable trajectories from well-known chaotic systems, **Lorenz**, **Lorenz–Stenflo**, and **Rikitake**, and calibrating them to match statistical and dynamical properties of real financial markets, we aim to test whether chaos-based features can improve predictive power and trading robustness.

The core idea: **use chaos as a structured source of complexity** to extract meaningful non-linear features, train ML/DL models, and backtest algorithmic trading strategies.

---

## 🧪 Project Pipeline

### 1. **Chaotic Signal Simulation**
- Implement Lorenz, Lorenz–Stenflo, and Rikitake equations
- Calibrate parameters to match target market volatility and entropy
- Generate multi-dimensional time series (`x(t), y(t), z(t)`)
- Store chaos signals in structured CSV format

### 2. **Feature Engineering**
- Sliding-window extraction of chaos-based features:
  - Statistical (mean, std, skewness, kurtosis)
  - Complexity (Sample Entropy, Permutation Entropy)
  - Recurrence Quantification Analysis (RQA)
  - Fractal dimension, Hurst exponent
  - Spectral (FFT peaks, spectral centroid, energy bands)
- Normalization and optional PCA/ICA dimensionality reduction

### 3. **Market Data Preparation**
- Collect and clean **1 year of BTC, 1 year of ETH, 1 year of SHIB** (1-min market data)
- Compute log returns, realized volatility, additional market features
- Synchronize chaos-based features with market timestamps

### 4. **Model Training & Validation**
- Train supervised ML/DL models:
  - Baselines: Logistic Regression, XGBoost
  - Deep Models: LSTM, GRU, Transformer for time series
- Primary training on BTC, cross-validation on ETH, stress test on SHIB
- Evaluation metrics: Accuracy, Sharpe improvement

### 5. **Backtesting**
- Convert model predictions into trading signals (long-only or long/short)
- Backtest with realistic transaction costs
- Compare against benchmarks (Buy & Hold, naive momentum)
- KPIs: Sharpe ratio, Sortino ratio, Max Drawdown, Hit Ratio

---

## 📂 Repository Structure
project_root/  
├── data/  
│ ├── btc/  
│ │ └── btc_1h_2023_2024.csv  
│ ├── eth/  
│ │ └── eth_1h_2023_2024.csv   
│ └── chaotic_signals/  
│ ├── lorenz_xyz.csv  
│ ├── stenflo_xyz.csv  
│ └── rikitake_xyz.csv  
│  
├── features/  
│ ├── btc_features.csv  
│ ├── eth_features.csv  
│ └── shib_features.csv  
│  
├── notebooks/  
│ ├── 01_simulate_chaotic_signals.ipynb  
│ ├── 02_feature_engineering.ipynb  
│ ├── 03_prepare_market_data.ipynb  
│ ├── 04_model_training_and_validation.ipynb  
│ └── 05_backtest_strategy.ipynb  
│  
├── backtest/   
│ └── results_summary.csv   
│  
├── README.md  
└── requirements.txt  


---

## 📊 Data Sources
- **Market Data**: Binance API via [CCXT](https://github.com/ccxt/ccxt)
- **Chaotic Systems**: Generated using custom Python ODE solvers (`scipy.integrate.odeint`)

---

## 📈 Example Applications
- Research: testing chaos theory’s applicability to market prediction
- Model robustness analysis across different asset classes
- Stress-testing ML pipelines with non-trivial synthetic feature sources
- Feature fusion between chaos-based and market-based indicators

---

## 🛠 Tech Stack
- **Python**: 3.13+
- **Core Libraries**: NumPy, Pandas, SciPy, Scikit-learn
- **Time Series ML**: TensorFlow / PyTorch, XGBoost, LightGBM
- **Backtesting**: `backtrader`
- **Visualisation**: Matplotlib, Seaborn, Plotly

## 📜 License
This project is licensed under the MIT License.
