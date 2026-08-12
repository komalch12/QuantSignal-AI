# QuantSignal AI — Automated Test Execution Report

**Date**: 2026-08-12  
**Test Runner**: `pytest` 7.x  
**Environment**: Windows 11 (Python 3.11)  
**Target Application**: QuantSignal AI (Assignment 1 Complete Suite)

---

## 📊 Executive Summary

- **Total Unit & Integration Tests**: **100**
- **Passed**: **100**
- **Failed**: **0**
- **Errors**: **0**
- **Execution Time**: **~7.6 seconds**
- **Pass Rate**: **100.0%**

```bash
$ .venv\Scripts\python.exe -m pytest -q
........................................................................ [ 72%]
............................                                             [100%]
100 passed in 7.63s
```

---

## 🧪 Major Test Suite Categories

### 1. Exchange Traded Quantity (ETQ) Service (`tests/test_etq_service.py`)
- **Tests (11)**:
  - Exact 5m, 20m, 60m boundary inclusion/exclusion.
  - Multi-window tick isolation.
  - Empty trade trade data & unknown symbols (returns 0).
  - Deduplication via `trade_id` and `(timestamp, price, quantity)` hashing upon WebSocket reconnects.
  - Out-of-order tick arrival sorting.
  - Invalid/zero/negative/NaN quantity sanitization.
  - Timezone normalization (UTC, IST, offset-naive).
  - 1-minute OHLCV candle ingestion into ETQ windows.
  - `ETQSnapshot` and DataFrame export schema integrity.

### 2. Average LTP Service (`tests/test_average_price_service.py`)
- **Tests (12)**:
  - Exact 20m and 60m time window boundaries.
  - Values inside vs. outside lookback windows.
  - Empty datasets & unknown symbols (returns `0.0`).
  - Missing, null, NaN, Inf price handling.
  - Invalid prices (`price <= 0.0`).
  - Deduplication of identical price observations.
  - Non-chronological tick arrival sorting.
  - Timezone handling across naive, UTC, and IST timestamps.
  - Known price series with deterministic expected averages (`[100, 110, 120] -> 120.0`).
  - Snapshot & DataFrame export schema integrity.

### 3. Crossover Profitability Evaluation (`tests/test_crossover_profitability.py`)
- **Tests (10)**:
  - Profitable BUY trade outcome (`Exit > Entry` $\rightarrow$ `PROFITABLE`, `PnL > 0`, `Return % > 0`).
  - Unprofitable BUY trade outcome (`Exit <= Entry` $\rightarrow$ `UNPROFITABLE`).
  - Profitable Short SELL trade outcome (`Exit < Entry` $\rightarrow$ `PROFITABLE`).
  - Unprofitable Short SELL trade outcome (`Exit >= Entry` $\rightarrow$ `UNPROFITABLE`).
  - Exact crossover entry price mapping at bar $k$.
  - Exit price evaluation after 5-bar holding horizon ($k + 5$).
  - Insufficient future data handling (signals in latest 4 bars $\rightarrow$ `INSUFFICIENT_DATA`).
  - Missing & invalid price sanitization.
  - BUY, SELL, and Overall Win Rate % calculations and sample trade counts.
  - Verification of Zero Look-Ahead Bias (signal generation at bar $k$ uses only history $\le k$).

### 4. Explicit ACCEPT / AVOID Trade Decision Layer (`tests/test_trade_decision_service.py`)
- **Tests (9)**:
  - Strong BUY signal $\rightarrow$ `ACCEPT` when AI confidence $\ge 60\%$, recommendation aligns, and win rate $\ge 50\%$.
  - Weak BUY signal (AI confidence $< 60\%$) $\rightarrow$ `AVOID`.
  - Strong SELL signal $\rightarrow$ `ACCEPT` when criteria satisfied.
  - Weak SELL signal (AI confidence $< 60\%$) $\rightarrow$ `AVOID`.
  - Conflicting AI recommendation (e.g. BUY crossover + SELL AI recommendation) $\rightarrow$ `AVOID`.
  - Poor historical win rate ($< 50\%$) $\rightarrow$ `AVOID`.
  - Strong historical evidence + high confidence $\rightarrow$ `ACCEPT`.
  - Missing / insufficient data $\rightarrow$ `AVOID`.
  - Verification that every `AVOID` decision populates a non-empty `decision_reason`.

### 5. Technical Indicators & SMMA 20/120 (`tests/test_technical_indicators.py`)
- **Tests (15)**:
  - SMMA 20 & SMMA 120 calculation accuracy against reference values.
  - Trend determination (`Bullish 🟢` vs `Bearish 🔴`).
  - Distance % between LTP and SMMA lines.
  - Market data provider integration.

### 6. SMMA Crossover Detection (`tests/test_crossover.py`, `tests/test_historical_crossover.py`)
- **Tests (25)**:
  - BUY crossover detection (`SMMA20` crossing above `SMMA120`).
  - SELL crossover detection (`SMMA20` crossing below `SMMA120`).
  - Robust scanning over 250 days of daily/intraday history.
  - Summary metrics cards calculation.

### 7. AI/ML Signal Engine (`tests/test_ml_signal.py`)
- **Tests (18)**:
  - Scikit-Learn Random Forest model initialization & loading (`quant_signal_rf_v1.joblib`).
  - Feature extraction (`rsi_14`, `macd`, `macd_signal`, `smma_20_ratio`, `smma_120_ratio`, `volatility_20`).
  - Probability inference (`confidence_pct`).
  - Model recommendation mapping (`STRONG BUY`, `BUY`, `HOLD`, `SELL`).
  - Feature importance weights extraction.

---

## 🔒 Security & Data Integrity Verification

- **Hard-coded Credentials**: **None**.
- **Secret Files**: `.env`, `access_token.txt`, `refresh_token.txt`, and `saved_tokens/` are excluded via `.gitignore`.
- **Placeholder Templates**: `.env.example` contains template placeholders only.

---

## ⚠️ Known Limitations

1. **Live FYERS Verification**:
   - Live broker tick streaming and WebSocket integration could not be verified because live broker credentials (`FYERS_CLIENT_ID` / `FYERS_SECRET_KEY`) were not configured in `.env`.
   - The application safely falls back to `DEVELOPMENT / DEMO` mode and clearly labels all data as simulated.
