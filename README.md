# Chaotic Dynamics as a Framework for Modelling Cryptocurrency Market Fluctuations

## 📌 Overview

This project investigates whether the price fluctuations of major cryptocurrency
assets (BTC, ETH) exhibit signatures consistent with low-dimensional chaotic
deterministic dynamics — and whether this structure can be exploited for
short-horizon prediction.

Chaotic systems are treated as **theoretical references** here: Lorenz, Lorenz–Stenflo, and Rikitake are simulated,
characterised, and used as benchmarks against which the empirical dynamics of crypto markets are measured.

The core pipeline moves from **theory → empirical characterisation → prediction**:
1. Simulate and characterise known chaotic systems (λ₁, D₂, Fourier spectrum)
2. Reconstruct the crypto market attractor via Takens delay embedding
3. Measure and compare chaotic indicators across assets
4. Train lightweight ML models on phase-space features and analyse the
   Lyapunov predictability horizon

---

## 🧪 Project Pipeline

### 1. Chaotic System Simulation
- Lorenz, Lorenz–Stenflo, Rikitake — ODE simulation (Euler method)
- Calibration: largest Lyapunov exponent (λ₁), correlation dimension (D₂), Fourier power spectrum
- Constitutes the **theoretical reference table** for comparison

### 2. Phase-Space Reconstruction (Takens Embedding)
- Load 1-minute BTC and ETH data (Binance historical archive, 2025)
- Estimate optimal delay τ via Average Mutual Information (AMI)
- Estimate embedding dimension m via False Nearest Neighbours (FNN)
- Reconstruct and visualise the market attractor in delay-coordinate space

### 3. Empirical Chaotic Characterisation
- Compute λ₁ (Rosenstein method), D₂ (Grassberger–Procaccia),
  permutation entropy, and BDS test on real crypto series
- Compare BTC, ETH, control asset (Gold / S&P 500), and theoretical systems
- Answer the core research question: **are crypto markets chaotic?**

### 4. Modelling & Prediction
- Features: delay embedding vectors from phase-space reconstruction
- Models: LSTM, XGBoost, Logistic Regression (baseline)
- Evaluation: accuracy, AUC, walk-forward validation
- Lyapunov horizon analysis: empirical predictability collapse vs theoretical T*

---

## 📂 Repository Structure
```
project_root/
├── data/
│   ├── raw/
│   │   ├── BTCUSDT_1m_2025.csv
│   │   └── ETHUSDT_1m_2025.csv
│   └── chaos/
│       ├── lorenz.parquet
│       ├── lorenz_stenflo.parquet
│       └── rikitake.parquet
│
├── figures/
│   ├── lorenz_attractor.png
│   ├── lorenz_stenflo_attractor.png
│   └── rikitake_attractor.png
│
├── notebooks/
│   ├── 01_simulate_chaotic_signals.ipynb
│   ├── 02_embedding.ipynb
│   ├── 03_characterisation.ipynb
│   ├── 04_modelling.ipynb
│   └── 05_lyapunov_horizon.ipynb  (optional)
│
├── README.md
└── requirements.txt
```

---

## 📊 Data Sources
- **Market Data**: Binance historical archive (1-minute OHLCV, 2025)
- **Chaotic Systems**: Custom Python ODE solvers (Euler method)

---

## 🛠 Tech Stack
- **Python** 3.13+
- **Core**: NumPy, Pandas, SciPy
- **ML**: Scikit-learn, XGBoost, TensorFlow / PyTorch
- **Nonlinear analysis**: nolds, antropy
- **Visualisation**: Matplotlib, Seaborn

---

## 📜 License
MIT License
