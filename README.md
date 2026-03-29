---
title: P2-ETF-Predictor
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
python_version: "3.11"
---
p2-etf-tft-outputs/
├── option_a/
│   └── global_model/               # Global model artifacts (Option A)
│       ├── GLD.weights.h5
│       ├── ... (other ETFs)
│       ├── scaler.pkl
│       ├── meta.json
│       └── signals.json
├── option_b/
│   └── global_model/               # Global model artifacts (Option B)
│       ├── SPY.weights.h5
│       ├── ...
│       ├── scaler.pkl
│       ├── meta.json
│       └── signals.json
├── sweep/                          # Per‑year model sweep results (for "Year Model" approach)
│   ├── option_a/
│   │   └── signals_{year}_{date}.json   (e.g., signals_2008_20260329.json)
│   └── option_b/
│       └── signals_{year}_{date}.json
└── global_sweep/                   # Global model sweep results (for "Global Model" approach)
    ├── option_a/
    │   └── signals_{year}_{date}.json
    └── option_b/
        └── signals_{year}_{date}.json
