Feature Engineering & Data Pipelines
🔹 Part A – Feature Engineering (Core ML Skill)

1. Data Cleaning

BTC-USD (yfinance) cleaning steps:

Normalize columns (Open→open, etc.) and sort by date

Drop duplicate timestamps and rows with all-NaN OHLCV

Handle missing prices with forward/backward fill; keep volume NaNs as 0

Compute log returns from close

Detect outliers on returns via z-score (|z| > 5) and clip to ±5σ

Save raw and cleaned CSVs under data/raw and data/processed

Note: Categorical encoding is not applicable for OHLCV time series
Data leakage examples (VERY important)

2. Feature Transformation

Scaling (StandardScaler vs MinMaxScaler)

Log transforms

Polynomial features

Binning

Date-time feature extraction

year, month, weekday

lag features (important for trading/time series)

3. Advanced Feature Engineering

Target encoding

Feature selection

Correlation filtering

Mutual Information

SHAP-based importance

Domain-driven features (finance, IoT sensor signals etc.)
