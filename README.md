# QuantSignal AI 🚀 — Assignment 1 Complete Suite

QuantSignal AI is a production-grade, modular quantitative trading signal engine and real-time dashboard framework built with **Python 3.11/3.12**, **Streamlit**, **Fyers API v3**, **Scikit-Learn**, **Pandas**, **NumPy**, and **Joblib**.

---

## 📋 Assignment 1 Requirement Compliance Matrix (18/18 COMPLETE)

| # | Requirement Description | Implementation Status | Core Module / File |
|---|-------------------------|-----------------------|--------------------|
| **1** | NSE Stocks with LTP ₹30 – ₹500 | **COMPLETE** | [`quant_signal/services/stock_scanner.py`](file:///c:/QuantSignal%20AI/quant_signal/services/stock_scanner.py) |
| **2** | Bid Quantity > 10,00,000 | **COMPLETE** | [`quant_signal/services/liquidity_filter.py`](file:///c:/QuantSignal%20AI/quant_signal/services/liquidity_filter.py) |
| **3** | Ask Quantity > 10,00,000 | **COMPLETE** | [`quant_signal/services/liquidity_filter.py`](file:///c:/QuantSignal%20AI/quant_signal/services/liquidity_filter.py) |
| **4** | SMMA 20 (Smoothed Moving Average) | **COMPLETE** | [`quant_signal/indicators/smma.py`](file:///c:/QuantSignal%20AI/quant_signal/indicators/smma.py) |
| **5** | SMMA 120 (Smoothed Moving Average) | **COMPLETE** | [`quant_signal/indicators/smma.py`](file:///c:/QuantSignal%20AI/quant_signal/indicators/smma.py) |
| **6** | Exchange Traded Quantity (5m, 20m, 60m) | **COMPLETE** | [`quant_signal/services/etq_service.py`](file:///c:/QuantSignal%20AI/quant_signal/services/etq_service.py) |
| **7** | Average LTP (20m, 60m) | **COMPLETE** | [`quant_signal/services/average_price_service.py`](file:///c:/QuantSignal%20AI/quant_signal/services/average_price_service.py) |
| **8** | Live Market Depth (Bid/Ask Prices & Quantities) | **COMPLETE** | [`quant_signal/services/market_depth_service.py`](file:///c:/QuantSignal%20AI/quant_signal/services/market_depth_service.py) |
| **9** | Real-time Tabular Dashboard | **COMPLETE** | [`quant_signal/ui/views.py`](file:///c:/QuantSignal%20AI/quant_signal/ui/views.py) |
| **10** | Automatic Real-time Refresh (1s) | **COMPLETE** | [`quant_signal/ui/views.py`](file:///c:/QuantSignal%20AI/quant_signal/ui/views.py) |
| **11** | SMMA Crossover Detection | **COMPLETE** | [`quant_signal/services/crossover_service.py`](file:///c:/QuantSignal%20AI/quant_signal/services/crossover_service.py) |
| **12** | BUY Crossover (SMMA20 crosses above SMMA120) | **COMPLETE** | [`quant_signal/services/crossover_service.py`](file:///c:/QuantSignal%20AI/quant_signal/services/crossover_service.py) |
| **13** | SELL Crossover (SMMA20 crosses below SMMA120) | **COMPLETE** | [`quant_signal/services/crossover_service.py`](file:///c:/QuantSignal%20AI/quant_signal/services/crossover_service.py) |
| **14** | AI/ML Signal Ranking & Predictions | **COMPLETE** | [`quant_signal/services/ml_signal_service.py`](file:///c:/QuantSignal%20AI/quant_signal/services/ml_signal_service.py) |
| **15** | Historical Profitability Evaluation | **COMPLETE** | [`quant_signal/services/crossover_profitability_service.py`](file:///c:/QuantSignal%20AI/quant_signal/services/crossover_profitability_service.py) |
| **16** | Explicit ACCEPT / AVOID Decision Layer | **COMPLETE** | [`quant_signal/services/crossover_profitability_service.py`](file:///c:/QuantSignal%20AI/quant_signal/services/crossover_profitability_service.py) |
| **17** | AI Model Confidence % & Probabilities | **COMPLETE** | [`quant_signal/services/ml_signal_service.py`](file:///c:/QuantSignal%20AI/quant_signal/services/ml_signal_service.py) |
| **18** | Human-Readable Avoidance Explanations | **COMPLETE** | [`quant_signal/services/crossover_profitability_service.py`](file:///c:/QuantSignal%20AI/quant_signal/services/crossover_profitability_service.py) |

---

## 🏛️ System Architecture

```
QuantSignal AI/
├── .env.example                                  # Environment variables template
├── .gitignore                                    # Git security & build exclusion rules
├── requirements.txt                              # Project dependencies
├── README.md                                     # Main documentation
├── app.py                                        # Streamlit web application entry point
├── saved_models/                                 # Trained Scikit-Learn RF model artifacts
│   └── quant_signal_rf_v1.joblib
├── documentation/                                # Reports and audit documentation
│   └── TEST_REPORT.md
├── quant_signal/                                 # Main Python Package
│   ├── core/                                     # Types, Data Classes & Base Interfaces
│   │   ├── types.py                              # Signal, Crossover & Profitability dataclasses
│   │   └── base.py                               # Abstract base contracts
│   ├── config/                                   # Configuration & Settings Management
│   │   └── settings.py
│   ├── providers/                                # Market Data Providers (Fyers & Demo)
│   │   ├── base.py
│   │   ├── demo_provider.py
│   │   └── fyers_provider.py
│   ├── services/                                 # Domain Microservices
│   │   ├── stock_scanner.py                      # Universe scanner (₹30-₹500 filter)
│   │   ├── liquidity_filter.py                   # Bid/Ask Qty > 1M filter
│   │   ├── technical_indicators.py               # Technical indicator calculator
│   │   ├── crossover_service.py                  # SMMA 20/120 crossover detector
│   │   ├── etq_service.py                        # Exchange Traded Quantity (5m/20m/60m)
│   │   ├── average_price_service.py              # Average LTP (20m/60m)
│   │   ├── crossover_profitability_service.py    # Trade evaluation & ACCEPT/AVOID policy
│   │   └── ml_signal_service.py                  # Scikit-Learn RF inference engine
│   └── ui/                                       # Streamlit Presentation Layer
│       ├── components.py                         # Reusable UI widgets & cards
│       └── views.py                              # View layouts & tabs
└── tests/                                        # Automated Pytest Suite (100/100 PASS)
    ├── test_technical_indicators.py
    ├── test_crossover.py
    ├── test_historical_crossover.py
    ├── test_ml_signal.py
    ├── test_etq_service.py
    ├── test_average_price_service.py
    ├── test_crossover_profitability.py
    └── test_trade_decision_service.py
```

---

## 🧮 Domain Methodologies

### 1. SMMA 20 / SMMA 120 Indicator Methodology
The Smoothed Moving Average (SMMA) removes noise while maintaining sensitivity to major trend shifts:
$$\text{SMMA}_i = \frac{\text{SMMA}_{i-1} \times (N - 1) + \text{Close}_i}{N}$$
where $N=20$ for short-term trend and $N=120$ for structural trend.

### 2. Exchange Traded Quantity (ETQ) Methodology
`ExchangeTradedQuantityService` tracks volume executed at the exchange during rolling time windows:
- **ETQ 5m**: Total volume in $[T - 5\text{min}, T]$
- **ETQ 20m**: Total volume in $[T - 20\text{min}, T]$
- **ETQ 60m**: Total volume in $[T - 60\text{min}, T]$
Includes tick-level deduplication via `trade_id` and timestamp/price hashing to prevent double-counting across WebSocket reconnects.

### 3. Average LTP Methodology
`AveragePriceService` calculates rolling price averages:
$$\text{Avg LTP}_{\text{window}} = \frac{\sum_{i=1}^{M} \text{LTP}_i}{M}$$
for $T_{\text{obs}} \in [T - \text{window}, T]$ over 20m and 60m periods.

### 4. Profitability Evaluation Methodology
Evaluates historical outcome of crossovers over a 5-bar evaluation horizon:
- **BUY Crossover**: $\text{PnL} = \text{Close}_{k+5} - \text{Close}_k$. Profitable if $\text{PnL} > 0$.
- **SELL Crossover**: $\text{PnL} = \text{Close}_k - \text{Close}_{k+5}$. Profitable if $\text{PnL} > 0$.
- **Zero Look-Ahead Bias**: Signal generation at bar $k$ uses history $\le k$. Future bars $k+1 \dots k+5$ are only used for historical outcome evaluation.

### 5. Explicit ACCEPT / AVOID Trade Decision Layer
Synthesizes 4 criteria into a deterministic execution signal:
- **ACCEPT** (`trade_decision = "ACCEPT"`):
  1. Data available and valid.
  2. AI Model Confidence $\ge 60.0\%$.
  3. ML recommendation aligns with crossover direction.
  4. Historical win rate $\ge 50.0\%$.
- **AVOID** (`trade_decision = "AVOID"`): Triggered if any criterion fails, producing an explicit human-readable `decision_reason`.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11 or 3.12
- Git

### Quickstart
```bash
# 1. Clone repo
git clone https://github.com/komalch12/QuantSignal-AI.git
cd QuantSignal-AI

# 2. Setup virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment template
cp .env.example .env

# 5. Launch dashboard
streamlit run app.py
```

---

## 🧪 Testing

Run full automated test suite (100 tests):
```bash
python -m pytest -q
```
**Output**: `100 passed in ~7.6s`.

---

## 🔒 Security Notes

- **Zero Hard-Coded Credentials**: Secrets are loaded dynamically from environment variables.
- **Git Protections**: `.env`, OAuth token files, virtual environments, and log files are excluded via `.gitignore`.
- **Placeholder Templates**: `.env.example` contains non-secret template strings only.

---

## ⚠️ Known Limitations

- **Live FYERS Verification**: Live tick streaming requires active FYERS API credentials in `.env`. When unconfigured, the system operates in **DEVELOPMENT / DEMO MODE** with sample data and non-misleading labels.
