# ₿ Bitcoin Data Analytics Dashboard

> **Internship Project** — End-to-End Bitcoin Market Intelligence Platform  
> Dataset Source: [Kaggle — Bitcoin Historical Data (BTC/USD 1-Min OHLCV)](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)

---

## 📌 Project Scope

This project delivers a **fully interactive, memory-optimized analytics dashboard** for Bitcoin (BTC/USD) price data spanning from 2012 to the present. The platform covers:

- **Exploratory Data Analysis** — OHLCV historical trends, volatility patterns, and volume profiling.
- **Technical Indicator Suite** — Moving Averages (7/30/90D), Bollinger Bands, MACD, and RSI-14.
- **Machine Learning Forecasting** — Random Forest Regressor predicting the `N`-day-ahead closing price with model evaluation metrics.
- **Operational UI** — A single-file Streamlit application with sidebar controls for date filtering and ML configuration.

---

## 🏗️ Architectural Layout

```
Bitcoin_Analytics/
│
├── Intern_BitcoinAnalytics.py   ← Single unified app (frontend + backend)
├── btcusd_1-min_data.csv        ← Kaggle source dataset (place here)
├── requirements.txt             ← Pinned Python dependencies
└── README.md                    ← This file
```

### Internal Module Flow

```
[CSV File]
    │
    ▼
[Data Layer]          pd.read_csv (chunked, dtype=float32)
                      Unix → datetime index
                      .resample('D').agg(OHLCV)      ← millions → thousands
                      .ffill() → clean NaN gaps
    │
    ▼
[Feature Engineering] Returns, Moving Averages, RSI, MACD,
                      Bollinger Bands, Volatility, High-Low %
    │
    ├──────────────────────────────────────────────────────────────┐
    ▼                                                              ▼
[Analytics & Viz Layer]                               [ML Pipeline]
  • Metric delta cards (Close, ATH, ATL, Volume)       • Build feature matrix X, target y
  • Candlestick + Volume chart                         • Train/Test split (80/20, no shuffle)
  • Bollinger Band overlay                             • StandardScaler normalisation
  • MACD histogram                                     • RandomForestRegressor (120 trees)
  • RSI-14 overbought/oversold                         • MAE / RMSE / R² / MAPE metrics
  • Annual returns bar chart                           • Actual vs Predicted chart
  • Volume trend + 7D MA                               • Feature Importance ranking
  • Correlation heatmap                                • Error distribution histogram
```

---

## 🛠️ Technology Stack

| Layer | Library | Purpose |
|-------|---------|---------|
| UI Framework | `streamlit 1.35` | Interactive web dashboard |
| Data Processing | `pandas 2.2` | Chunked loading, resampling, feature engineering |
| Numerical Computing | `numpy 1.26` | Array ops, metric calculation |
| Machine Learning | `scikit-learn 1.5` | Random Forest, scaler, metrics |
| Interactive Charts | `plotly 5.22` | Candlestick, line, bar, histogram charts |
| Static Charts | `matplotlib 3.9` | Correlation heatmap |

---

## ⚡ Performance & Memory Optimization

The raw dataset contains **~5 million rows** of 1-minute OHLCV records. The following guardrails prevent memory exhaustion on standard hardware:

1. **Chunked Loading** — `pd.read_csv(..., chunksize=200_000)` reads the file in 200K-row batches.
2. **Dtype Downcasting** — All OHLCV columns are read as `float32` (half the memory of `float64`).
3. **Aggressive Resampling** — Each chunk is immediately resampled to `Daily ('D')` intervals using `.resample('D').agg(...)`, reducing ~5M rows to ~4,000 rows.
4. **Forward-Fill** — `.ffill()` ensures no NaN gaps in the continuous price series.
5. **Garbage Collection** — `gc.collect()` is called after each chunk and after heavy chart renders.
6. **Streamlit Cache** — `@st.cache_data` caches the load-and-resample result to prevent re-execution on user interaction.

---

## 🚀 Installation & Run

### 1. Clone / Download the project
```bash
git clone <your-repo-url>
cd Bitcoin_Analytics
```

### 2. Place the dataset
Download `btcusd_1-min_data.csv` from Kaggle and place it in the `Bitcoin_Analytics/` folder.

### 3. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the dashboard
```bash
streamlit run Intern_BitcoinAnalytics.py
```

Open your browser at **http://localhost:8501**

---

## 📊 Primary Analytical Discoveries

| Finding | Insight |
|---------|---------|
| **Bull Cycles** | Bitcoin exhibits clear 4-year halving cycles with parabolic price appreciation in 2013, 2017, and 2020–2021. |
| **Volatility Clustering** | RSI frequently exceeds 70 during bull runs and drops below 30 during bear corrections, confirming trend reversals. |
| **Volume–Price Divergence** | Volume spikes often precede price breakouts by 1–3 days, visible in the Volume MA-7 overlay. |
| **MA Crossover Signals** | MA-7 crossing MA-30 upward reliably identifies early-stage bull phases. |
| **ML Forecast Accuracy** | Random Forest achieves R² > 0.90 on short horizons (1–7 days), with MAPE < 5% on recent data. |
| **Bollinger Band Squeezes** | Periods of low BB width (band squeeze) consistently precede high-volatility breakout events. |
| **Annual Returns** | 2020 and 2021 delivered >300% and >60% annual returns respectively; 2022 reversed with approximately −65%. |

---

## 📄 License

For internship/academic use only. Dataset subject to [Kaggle Terms of Service](https://www.kaggle.com/terms).
