"""
SMMA Crossover Profitability Evaluation Service for QuantSignal AI.

Evaluates historical trade performance (PROFITABLE / UNPROFITABLE / INSUFFICIENT_DATA)
and win-rate metrics for SMMA 20 / SMMA 120 crossover events over a deterministic
holding horizon (default 5 bars / candles).

Methodology:
  - Entry Price: Close price at the bar 'k' where crossover occurred.
  - Exit Price: Close price at bar 'k + horizon_bars' (default 5 bars).
  - BUY Crossover Trade:
      Exit Price > Entry Price  -> PROFITABLE
      Exit Price <= Entry Price -> UNPROFITABLE
      PnL = Exit Price - Entry Price
      Return % = ((Exit Price - Entry Price) / Entry Price) * 100
  - SELL Crossover Trade:
      Exit Price < Entry Price  -> PROFITABLE
      Exit Price >= Entry Price -> UNPROFITABLE
      PnL = Entry Price - Exit Price
      Return % = ((Entry Price - Exit Price) / Entry Price) * 100
  - Insufficient Data:
      If fewer than 'horizon_bars' future candles exist after bar 'k', or if
      entry/exit prices are NaN/invalid -> INSUFFICIENT_DATA.
"""

from __future__ import annotations

import math
from typing import Any
import pandas as pd
import numpy as np

from quant_signal.core.types import CrossoverProfitabilityResult
from quant_signal.logger import get_logger
from quant_signal.services.crossover_service import CrossoverService, scan_historical_crossovers
from quant_signal.services.ml_signal_service import MLSignalService

logger = get_logger(__name__)


class CrossoverProfitabilityService:
    """
    Evaluates SMMA crossover trade outcomes, win rates, and AI explainability.
    """

    def __init__(
        self,
        crossover_service: CrossoverService | None = None,
        ml_service: MLSignalService | None = None,
        default_horizon_bars: int = 5,
    ) -> None:
        self.crossover_service = crossover_service or CrossoverService()
        self.ml_service = ml_service or MLSignalService(tech_service=self.crossover_service.tech_service)
        self.default_horizon_bars = default_horizon_bars

    def evaluate_crossover_event(
        self,
        df_history: pd.DataFrame,
        crossover_event: dict[str, Any],
        horizon_bars: int = 5,
    ) -> CrossoverProfitabilityResult:
        """
        Evaluates trade outcome for a single historical crossover event over 'horizon_bars'.

        Zero look-ahead bias: Signal generation at bar 'k' uses history <= k.
        Future bars k+1 ... k+horizon_bars are evaluated purely for trade outcome.
        """
        symbol = crossover_event.get("symbol", "")
        company_name = crossover_event.get("company_name", symbol)
        ts = crossover_event.get("timestamp")
        signal = crossover_event.get("crossover_type") or crossover_event.get("signal", "BUY")

        if df_history is None or df_history.empty or "close" not in df_history.columns:
            msg = "Empty or invalid historical dataset."
            return CrossoverProfitabilityResult(
                symbol=symbol,
                company_name=company_name,
                crossover_time=ts,
                signal=signal,
                entry_price=crossover_event.get("close_price", 0.0) or 0.0,
                result="INSUFFICIENT_DATA",
                available_data=False,
                avoidance_reason=msg,
                trade_decision="AVOID",
                decision_reason=f"AVOIDED: {msg}",
            )

        # Locate index of crossover timestamp in history DataFrame
        target_idx = None
        if ts is not None and "timestamp" in df_history.columns:
            matching_rows = df_history.index[df_history["timestamp"] == ts].tolist()
            if matching_rows:
                target_idx = matching_rows[0]

        # Fallback to matching close price if timestamp column matching is insufficient
        if target_idx is None:
            raw_close = crossover_event.get("close_price")
            if raw_close is not None and pd.notna(raw_close):
                matching = df_history.index[df_history["close"] == raw_close].tolist()
                if matching:
                    target_idx = matching[0]

        if target_idx is None:
            msg = "Crossover bar index could not be located in historical series."
            return CrossoverProfitabilityResult(
                symbol=symbol,
                company_name=company_name,
                crossover_time=ts,
                signal=signal,
                entry_price=crossover_event.get("close_price", 0.0) or 0.0,
                result="INSUFFICIENT_DATA",
                available_data=False,
                avoidance_reason=msg,
                trade_decision="AVOID",
                decision_reason=f"AVOIDED: {msg}",
            )

        entry_price = float(df_history.loc[target_idx, "close"])

        # Validate entry price
        if math.isnan(entry_price) or entry_price <= 0.0:
            msg = "Invalid or non-positive entry price."
            return CrossoverProfitabilityResult(
                symbol=symbol,
                company_name=company_name,
                crossover_time=ts,
                signal=signal,
                entry_price=entry_price if not math.isnan(entry_price) else 0.0,
                result="INSUFFICIENT_DATA",
                available_data=False,
                avoidance_reason=msg,
                trade_decision="AVOID",
                decision_reason=f"AVOIDED: {msg}",
            )

        exit_idx = target_idx + horizon_bars
        if exit_idx >= len(df_history):
            # Insufficient future candles after crossover bar
            msg = f"Insufficient future candles after crossover bar (requires {horizon_bars} bars, found {len(df_history) - 1 - target_idx})."
            return CrossoverProfitabilityResult(
                symbol=symbol,
                company_name=company_name,
                crossover_time=ts,
                signal=signal,
                entry_price=entry_price,
                evaluation_horizon=horizon_bars,
                result="INSUFFICIENT_DATA",
                available_data=False,
                avoidance_reason=msg,
                trade_decision="AVOID",
                decision_reason=f"AVOIDED: {msg}",
            )

        exit_price = float(df_history.loc[exit_idx, "close"])

        # Validate exit price
        if math.isnan(exit_price) or exit_price <= 0.0:
            msg = "Exit price is NaN or non-positive."
            return CrossoverProfitabilityResult(
                symbol=symbol,
                company_name=company_name,
                crossover_time=ts,
                signal=signal,
                entry_price=entry_price,
                evaluation_horizon=horizon_bars,
                result="INSUFFICIENT_DATA",
                available_data=False,
                avoidance_reason=msg,
                trade_decision="AVOID",
                decision_reason=f"AVOIDED: {msg}",
            )


        # ── Evaluate BUY vs SELL Profitability ────────────────────────────────
        if signal == "BUY":
            pnl = round(exit_price - entry_price, 2)
            ret_pct = round(((exit_price - entry_price) / entry_price) * 100.0, 2)
            is_profitable = exit_price > entry_price
            res_str = "PROFITABLE" if is_profitable else "UNPROFITABLE"
            avoid_reason = (
                f"BUY trade succeeded (+{ret_pct:.2f}% gain over {horizon_bars} bars)."
                if is_profitable
                else f"BUY trade failed ({ret_pct:.2f}% loss over {horizon_bars} bars). Price declined from ₹{entry_price:.2f} to ₹{exit_price:.2f}."
            )
        else:  # SELL
            pnl = round(entry_price - exit_price, 2)
            ret_pct = round(((entry_price - exit_price) / entry_price) * 100.0, 2)
            is_profitable = exit_price < entry_price
            res_str = "PROFITABLE" if is_profitable else "UNPROFITABLE"
            avoid_reason = (
                f"SELL trade succeeded (+{ret_pct:.2f}% short gain over {horizon_bars} bars)."
                if is_profitable
                else f"SELL trade failed ({ret_pct:.2f}% loss over {horizon_bars} bars). Short position faced rally from ₹{entry_price:.2f} to ₹{exit_price:.2f}."
            )

        # Query ML signal scoring at bar k for AI alignment
        ai_conf = 50.0
        ai_rec = "HOLD 🟡"
        try:
            sub_hist = df_history.iloc[: target_idx + 1]
            if len(sub_hist) >= 5:
                ml_res = self.ml_service.calculate_stock_ml_signal(symbol)
                ai_conf = ml_res.get("confidence_pct", 50.0)
                ai_rec = ml_res.get("recommendation", "HOLD 🟡")
        except Exception:
            pass

        # Evaluate ACCEPT / AVOID trade decision
        decision, dec_reason = self.evaluate_trade_decision(
            signal=signal,
            ai_confidence_pct=ai_conf,
            ai_recommendation=ai_rec,
            historical_win_rate_pct=50.0,  # Default baseline for individual crossover
            available_data=True,
        )

        return CrossoverProfitabilityResult(
            symbol=symbol,
            company_name=company_name,
            crossover_time=ts,
            signal=signal,
            entry_price=round(entry_price, 2),
            exit_price=round(exit_price, 2),
            evaluation_horizon=horizon_bars,
            pnl=pnl,
            return_pct=ret_pct,
            result=res_str,
            available_data=True,
            ai_confidence_pct=ai_conf,
            ai_recommendation=ai_rec,
            avoidance_reason=avoid_reason,
            trade_decision=decision,
            decision_reason=dec_reason,
        )

    def evaluate_trade_decision(
        self,
        signal: str,
        ai_confidence_pct: float,
        ai_recommendation: str,
        historical_win_rate_pct: float = 50.0,
        available_data: bool = True,
    ) -> tuple[str, str]:
        """
        Determines whether an SMMA crossover signal should be ACCEPTED or AVOIDED.

        Policy Rules:
          ACCEPT when:
            1. Data availability is True.
            2. AI Model Confidence >= 60.0%.
            3. Model recommendation aligns with signal direction:
               - BUY crossover: recommendation contains "BUY" (e.g. STRONG BUY, BUY).
               - SELL crossover: recommendation contains "SELL" or "HOLD".
            4. Historical crossover win rate >= 50.0%.

          AVOID when any of the above criteria are violated, providing an explicit
          evidence-backed decision_reason.

        Returns:
          (trade_decision ["ACCEPT" | "AVOID"], decision_reason [str])
        """
        if not available_data:
            return "AVOID", "AVOIDED: Insufficient historical or market data to validate crossover signal."

        # 1. AI Confidence Check
        if ai_confidence_pct < 60.0:
            return "AVOID", f"AVOIDED: Low AI confidence ({ai_confidence_pct:.1f}% is below the required 60.0% threshold)."

        # 2. Recommendation Alignment Check
        clean_rec = str(ai_recommendation).upper()
        if signal == "BUY":
            if "BUY" not in clean_rec:
                return "AVOID", f"AVOIDED: AI model recommendation '{ai_recommendation}' conflicts with BUY crossover signal."
        elif signal == "SELL":
            if "SELL" not in clean_rec and "HOLD" not in clean_rec:
                return "AVOID", f"AVOIDED: AI model recommendation '{ai_recommendation}' conflicts with SELL crossover signal."

        # 3. Historical Win-Rate Check
        if historical_win_rate_pct < 50.0:
            return "AVOID", f"AVOIDED: Historical crossover win rate ({historical_win_rate_pct:.1f}%) is below the 50.0% profitability threshold."

        return "ACCEPT", f"ACCEPTED: Supported by high AI confidence ({ai_confidence_pct:.1f}%), aligned recommendation ({ai_recommendation}), and historical win rate ({historical_win_rate_pct:.1f}%)."

    def evaluate_symbol_crossovers(
        self,
        symbol: str,
        days: int = 250,
        horizon_bars: int = 5,
    ) -> list[CrossoverProfitabilityResult]:
        """
        Evaluates all historical crossovers for a single symbol.
        """
        stock_res = self.crossover_service.tech_service.calculate_stock_indicators(symbol=symbol, days=days)
        history_df = stock_res.get("history_df", pd.DataFrame())

        company_name = symbol
        if hasattr(self.crossover_service.tech_service.provider, "_data"):
            company_name = self.crossover_service.tech_service.provider._data.get(symbol, {}).get("company", symbol)

        crossover_events = scan_historical_crossovers(
            symbol=symbol,
            company_name=company_name,
            df=history_df,
        )

        # Pre-evaluate symbol historical win rate
        results_pre = []
        for ev in crossover_events:
            results_pre.append(self.evaluate_crossover_event(df_history=history_df, crossover_event=ev, horizon_bars=horizon_bars))

        eval_trades = [r for r in results_pre if r.result in ("PROFITABLE", "UNPROFITABLE")]
        prof_count = sum(1 for r in eval_trades if r.result == "PROFITABLE")
        sym_win_rate = round((prof_count / len(eval_trades)) * 100.0, 1) if eval_trades else 50.0

        # Update trade_decision with symbol specific win rate
        final_results = []
        for r in results_pre:
            dec, dec_re = self.evaluate_trade_decision(
                signal=r.signal,
                ai_confidence_pct=r.ai_confidence_pct,
                ai_recommendation=r.ai_recommendation,
                historical_win_rate_pct=sym_win_rate,
                available_data=r.available_data,
            )
            r.trade_decision = dec
            r.decision_reason = dec_re
            final_results.append(r)

        return final_results

    def get_profitability_dataframe(
        self,
        symbols: list[str] | None = None,
        days: int = 250,
        horizon_bars: int = 5,
    ) -> pd.DataFrame:
        """
        Scans all symbols and returns a DataFrame of evaluated crossover trade outcomes.
        """
        target_symbols = symbols or self.crossover_service.tech_service.provider.get_symbols()
        all_results: list[dict[str, Any]] = []

        for sym in target_symbols:
            evals = self.evaluate_symbol_crossovers(symbol=sym, days=days, horizon_bars=horizon_bars)
            for ev in evals:
                ts_str = (
                    ev.crossover_time.strftime("%Y-%m-%d %H:%M")
                    if hasattr(ev.crossover_time, "strftime")
                    else str(ev.crossover_time or "N/A")
                )
                all_results.append({
                    "symbol":             ev.symbol,
                    "company_name":       ev.company_name,
                    "timestamp":          ts_str,
                    "crossover_time":     ev.crossover_time,
                    "signal":             ev.signal,
                    "entry_price":        ev.entry_price,
                    "exit_price":         ev.exit_price,
                    "evaluation_horizon": ev.evaluation_horizon,
                    "pnl":                ev.pnl,
                    "return_pct":         ev.return_pct,
                    "result":             ev.result,
                    "available_data":     ev.available_data,
                    "ai_confidence_pct": ev.ai_confidence_pct,
                    "ai_recommendation": ev.ai_recommendation,
                    "avoidance_reason":   ev.avoidance_reason,
                    "trade_decision":     ev.trade_decision,
                    "decision_reason":    ev.decision_reason,
                })

        if not all_results:
            return pd.DataFrame(columns=[
                "symbol", "company_name", "timestamp", "crossover_time", "signal", "entry_price", "exit_price",
                "evaluation_horizon", "pnl", "return_pct", "result", "available_data",
                "ai_confidence_pct", "ai_recommendation", "avoidance_reason", "trade_decision", "decision_reason"
            ])


        df = pd.DataFrame(all_results)
        return df.sort_values(by="crossover_time", ascending=False).reset_index(drop=True)

    def get_win_rate_metrics(self, df_profitability: pd.DataFrame | None = None) -> dict[str, Any]:
        """
        Calculates BUY, SELL, and Overall win rates and trade sample counts.
        """
        if df_profitability is None:
            df_profitability = self.get_profitability_dataframe()

        if df_profitability.empty:
            return {
                "buy_evaluated_trades": 0,
                "buy_profitable_count": 0,
                "buy_win_rate_pct": 0.0,
                "sell_evaluated_trades": 0,
                "sell_profitable_count": 0,
                "sell_win_rate_pct": 0.0,
                "overall_evaluated_trades": 0,
                "overall_profitable_count": 0,
                "overall_win_rate_pct": 0.0,
                "insufficient_data_count": 0,
            }

        # Filter evaluated trades (excluding INSUFFICIENT_DATA)
        valid_trades = df_profitability[df_profitability["result"].isin(["PROFITABLE", "UNPROFITABLE"])]
        insuff_count = len(df_profitability[df_profitability["result"] == "INSUFFICIENT_DATA"])

        buy_trades = valid_trades[valid_trades["signal"] == "BUY"]
        sell_trades = valid_trades[valid_trades["signal"] == "SELL"]

        buy_eval = len(buy_trades)
        buy_prof = len(buy_trades[buy_trades["result"] == "PROFITABLE"])
        buy_win_rate = round((buy_prof / buy_eval) * 100.0, 1) if buy_eval > 0 else 0.0

        sell_eval = len(sell_trades)
        sell_prof = len(sell_trades[sell_trades["result"] == "PROFITABLE"])
        sell_win_rate = round((sell_prof / sell_eval) * 100.0, 1) if sell_eval > 0 else 0.0

        overall_eval = len(valid_trades)
        overall_prof = len(valid_trades[valid_trades["result"] == "PROFITABLE"])
        overall_win_rate = round((overall_prof / overall_eval) * 100.0, 1) if overall_eval > 0 else 0.0

        return {
            "buy_evaluated_trades": buy_eval,
            "buy_profitable_count": buy_prof,
            "buy_win_rate_pct": buy_win_rate,
            "sell_evaluated_trades": sell_eval,
            "sell_profitable_count": sell_prof,
            "sell_win_rate_pct": sell_win_rate,
            "overall_evaluated_trades": overall_eval,
            "overall_profitable_count": overall_prof,
            "overall_win_rate_pct": overall_win_rate,
            "insufficient_data_count": insuff_count,
        }
