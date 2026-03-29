# config.py — ETF universes and shared parameters for P2-ETF-TFT-PREDICTOR-HF

# Option A: Fixed Income / Commodities (existing)
OPTION_A_ETFS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]

# Option B: Equity Sectors (new)
# Note: SPY is excluded because it's used as the benchmark
OPTION_B_ETFS = [
    "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLU", "GDX", "XME"
]

# Benchmarks (shared)
BENCHMARKS = ["SPY", "AGG"]

# Combined list for data fetching (all tickers)
ALL_TICKERS = OPTION_A_ETFS + OPTION_B_ETFS + BENCHMARKS# still includes SPY/AGG
