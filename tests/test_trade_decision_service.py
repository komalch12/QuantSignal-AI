"""
Unit tests for explicit ACCEPT / AVOID Trade Decision Layer in CrossoverProfitabilityService.

Verifies:
  1. Strong BUY signal -> ACCEPT when criteria satisfied.
  2. Weak BUY signal (confidence < 60%) -> AVOID.
  3. Strong SELL signal -> ACCEPT when criteria satisfied.
  4. Weak SELL signal (confidence < 60%) -> AVOID.
  5. Conflicting model recommendation -> AVOID.
  6. Poor historical win rate (< 50%) -> AVOID.
  7. Strong historical evidence + sufficient confidence -> ACCEPT.
  8. Missing / insufficient data -> AVOID handled cleanly.
  9. Every AVOID decision has a clear decision_reason.
  10. Preserves existing AI recommendation labels (STRONG BUY, BUY, HOLD, SELL).
"""

from __future__ import annotations

import pytest
from quant_signal.services.crossover_profitability_service import CrossoverProfitabilityService


@pytest.fixture
def prof_service() -> CrossoverProfitabilityService:
    return CrossoverProfitabilityService()


def test_strong_buy_accept(prof_service: CrossoverProfitabilityService) -> None:
    dec, reason = prof_service.evaluate_trade_decision(
        signal="BUY",
        ai_confidence_pct=78.5,
        ai_recommendation="STRONG BUY 🟢",
        historical_win_rate_pct=65.0,
        available_data=True,
    )
    assert dec == "ACCEPT"
    assert "ACCEPTED" in reason
    assert "78.5%" in reason


def test_weak_buy_avoid(prof_service: CrossoverProfitabilityService) -> None:
    dec, reason = prof_service.evaluate_trade_decision(
        signal="BUY",
        ai_confidence_pct=42.0,
        ai_recommendation="BUY 🟢",
        historical_win_rate_pct=65.0,
        available_data=True,
    )
    assert dec == "AVOID"
    assert "Low AI confidence" in reason
    assert "42.0%" in reason


def test_strong_sell_accept(prof_service: CrossoverProfitabilityService) -> None:
    dec, reason = prof_service.evaluate_trade_decision(
        signal="SELL",
        ai_confidence_pct=81.0,
        ai_recommendation="SELL 🔴",
        historical_win_rate_pct=58.0,
        available_data=True,
    )
    assert dec == "ACCEPT"
    assert "ACCEPTED" in reason


def test_weak_sell_avoid(prof_service: CrossoverProfitabilityService) -> None:
    dec, reason = prof_service.evaluate_trade_decision(
        signal="SELL",
        ai_confidence_pct=45.0,
        ai_recommendation="SELL 🔴",
        historical_win_rate_pct=58.0,
        available_data=True,
    )
    assert dec == "AVOID"
    assert "Low AI confidence" in reason


def test_conflicting_model_recommendation_avoid(prof_service: CrossoverProfitabilityService) -> None:
    # BUY crossover, but AI recommendation is SELL 🔴
    dec, reason = prof_service.evaluate_trade_decision(
        signal="BUY",
        ai_confidence_pct=75.0,
        ai_recommendation="SELL 🔴",
        historical_win_rate_pct=65.0,
        available_data=True,
    )
    assert dec == "AVOID"
    assert "conflicts with BUY crossover signal" in reason


def test_poor_historical_win_rate_avoid(prof_service: CrossoverProfitabilityService) -> None:
    # High confidence & aligned recommendation, but historical win rate is 35.0% (< 50%)
    dec, reason = prof_service.evaluate_trade_decision(
        signal="BUY",
        ai_confidence_pct=75.0,
        ai_recommendation="BUY 🟢",
        historical_win_rate_pct=35.0,
        available_data=True,
    )
    assert dec == "AVOID"
    assert "below the 50.0% profitability threshold" in reason


def test_strong_historical_evidence_accept(prof_service: CrossoverProfitabilityService) -> None:
    dec, reason = prof_service.evaluate_trade_decision(
        signal="BUY",
        ai_confidence_pct=62.0,
        ai_recommendation="BUY 🟢",
        historical_win_rate_pct=75.0,
        available_data=True,
    )
    assert dec == "ACCEPT"
    assert "ACCEPTED" in reason


def test_missing_data_avoid(prof_service: CrossoverProfitabilityService) -> None:
    dec, reason = prof_service.evaluate_trade_decision(
        signal="BUY",
        ai_confidence_pct=75.0,
        ai_recommendation="BUY 🟢",
        historical_win_rate_pct=65.0,
        available_data=False,
    )
    assert dec == "AVOID"
    assert "Insufficient historical or market data" in reason


def test_every_avoid_has_reason(prof_service: CrossoverProfitabilityService) -> None:
    test_cases = [
        ("BUY", 30.0, "BUY 🟢", 60.0, True),
        ("SELL", 70.0, "BUY 🟢", 60.0, True),
        ("BUY", 70.0, "BUY 🟢", 40.0, True),
        ("SELL", 70.0, "SELL 🔴", 60.0, False),
    ]
    for sig, conf, rec, wr, data in test_cases:
        dec, reason = prof_service.evaluate_trade_decision(sig, conf, rec, wr, data)
        assert dec == "AVOID"
        assert len(reason.strip()) > 0
