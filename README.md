# QuantSignal AI 🚀

A production-ready, highly scalable Python 3.12 quantitative trading signal engine and real-time dashboard framework built with **Streamlit**, **Fyers API v3**, **Scikit-learn**, **Pandas**, **NumPy**, **TA-Lib/ta**, and **Joblib**.

---

## 🏛️ Architecture Overview

QuantSignal AI is engineered using strict **SOLID principles**, modular package layout, strong type annotations, robust error handling, and structured logging.

```
QuantSignal AI/
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
├── requirements.txt          # Project dependencies (Python 3.12)
├── README.md                 # Project documentation
├── config.py                 # Core settings entry point
├── app.py                    # Streamlit web application entry point
└── quant_signal/             # Core Package Root
    ├── __init__.py
    ├── logger.py             # Structured logging system
    ├── exceptions.py         # Custom exception hierarchy
    ├── core/                 # Abstract contracts & type definitions
    │   ├── base.py           # Interface contracts (ABC)
    │   └── types.py          # Enums & Dataclasses (OHLCV, Signal, Order)
    ├── config/               # Pydantic Settings & Env loaders
    │   └── settings.py
    ├── brokers/              # Broker API Adapters (Fyers REST & WebSocket)
    │   ├── base.py
    │   └── fyers_adapter.py
    ├── indicators/           # Technical Analysis Engine
    │   ├── base.py
    │   └── ta_indicators.py
    ├── models/               # Machine Learning Model Management (Joblib/Sklearn)
    │   ├── base.py
    │   └── ml_model.py
    ├── strategy/             # Quantitative Strategy Orchestrator
    │   ├── base.py
    │   └── quant_strategy.py
    ├── ui/                   # Modular Streamlit UI Presentation Layer
    │   ├── components.py
    │   └── views.py
    └── utils/                # Helper utilities (Datetime, Math, Metrics)
        ├── datetime_utils.py
        ├── math_utils.py
        └── helpers.py
```

---

## 🧱 SOLID Architecture Highlights

1. **Single Responsibility Principle (SRP)**: Distinct separation between data retrieval, indicators, ML model scoring, signal logic, logging, and presentation.
2. **Open/Closed Principle (OCP)**: Abstract classes enable adding new brokers (e.g. Zerodha, IBKR) or new trading strategies without modifying existing codebase.
3. **Liskov Substitution Principle (LSP)**: All engine implementations strictly comply with defined interfaces.
4. **Interface Segregation Principle (ISP)**: Granular interfaces defined for Broker Adapters, Indicators, Models, and Strategies.
5. **Dependency Inversion Principle (DIP)**: High-level strategy engine depends on abstractions, enabling offline backtesting and paper trading mocks.

---

## 🛠️ Requirements & Installation

### Prerequisites
- **Python 3.12+**
- Fyers API v3 App Account Credentials

### Installation
```bash
# 1. Clone or navigate to the project directory
cd "c:/QuantSignal AI"

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment configuration
cp .env.example .env
```

---

## 🚀 Running the Application

```bash
streamlit run app.py
```

---

## 📜 License

MIT License. Designed for quantitative software architects and traders.
