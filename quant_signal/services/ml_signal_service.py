"""
AI / Machine Learning Signal Scoring Engine for QuantSignal AI.

Phase 7 Implementation:
- Feature extraction from OHLCV and SMMA 20/120 technical indicators.
- Scikit-Learn Model inference & probability scoring.
- Explainable feature importance scoring.
- Broker-independent operation in DEVELOPMENT / DEMO DATA mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from quant_signal.logger import get_logger
from quant_signal.services.technical_indicators import TechnicalIndicatorService
from quant_signal.services.crossover_service import CrossoverService

logger = get_logger(__name__)


def extract_features_from_history(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts deterministic ML feature vectors from stock historical DataFrame.

    Step 4A Features:
      1. distance_pct: ((smma_20 - smma_120) / smma_120) * 100.0
      2. smma_20_slope: 5-bar ROC of SMMA 20
      3. smma_120_slope: 5-bar ROC of SMMA 120
      4. price_smma20_dist: ((close - smma_20) / smma_20) * 100.0
      5. crossover_val: +1 for BUY crossover, -1 for SELL crossover, 0 for NO CROSSOVER
    """
    if history_df is None or history_df.empty or "smma_20" not in history_df.columns or "smma_120" not in history_df.columns:
        return pd.DataFrame()

    df = history_df.copy()
    valid_df = df.dropna(subset=["smma_20", "smma_120"]).copy()
    if len(valid_df) < 5:
        return pd.DataFrame()

    # 1. Distance percentage
    smma_120_safe = valid_df["smma_120"].replace(0, np.nan)
    valid_df["distance_pct"] = ((valid_df["smma_20"] - valid_df["smma_120"]) / smma_120_safe) * 100.0

    # 2 & 3. 5-bar slope / Rate of Change (ROC)
    valid_df["smma_20_slope"] = valid_df["smma_20"].pct_change(5) * 100.0
    valid_df["smma_120_slope"] = valid_df["smma_120"].pct_change(5) * 100.0

    # 4. Price/SMMA20 Spread %
    smma_20_safe = valid_df["smma_20"].replace(0, np.nan)
    close_price = valid_df["close"] if "close" in valid_df.columns else valid_df["smma_20"]
    valid_df["price_smma20_dist"] = ((close_price - valid_df["smma_20"]) / smma_20_safe) * 100.0

    # 5. Crossover Event Code (+1 = BUY, -1 = SELL, 0 = NO CROSSOVER)
    # Reuses exact Step 3 crossover detection rules
    prev_20 = valid_df["smma_20"].shift(1)
    prev_120 = valid_df["smma_120"].shift(1)
    curr_20 = valid_df["smma_20"]
    curr_120 = valid_df["smma_120"]

    crossover_code = np.zeros(len(valid_df), dtype=int)
    valid_prev = prev_20.notna() & prev_120.notna()
    buy_mask = (prev_20 <= prev_120) & (curr_20 > curr_120) & valid_prev
    sell_mask = (prev_20 >= prev_120) & (curr_20 < curr_120) & valid_prev

    crossover_code[buy_mask] = 1
    crossover_code[sell_mask] = -1
    valid_df["crossover_val"] = crossover_code

    feature_cols = ["distance_pct", "smma_20_slope", "smma_120_slope", "price_smma20_dist", "crossover_val"]
    feature_df = valid_df[feature_cols].fillna(0.0)
    return feature_df


def map_confidence_to_recommendation(conf_pct: float, crossover: str = "NONE") -> str:
    """
    Step 4B: Deterministic AI recommendation rule layer based on Random Forest confidence score
    and technical crossover alignment.

    Rules:
      - STRONG BUY 🟢: conf_pct >= 75.0 OR (crossover == "BUY_CROSSOVER" and conf_pct >= 60.0)
      - BUY 🟢:        conf_pct >= 60.0 OR crossover == "BUY_CROSSOVER"
      - SELL 🔴:       conf_pct <= 35.0 OR crossover == "SELL_CROSSOVER"
      - HOLD 🟡:       Otherwise (40% - 59.9% neutral confidence without technical crossover)

    Deterministic & safe against NaN / missing values.
    """
    if conf_pct is None or not np.isfinite(conf_pct):
        conf_pct = 50.0

    crossover_clean = crossover or "NONE"

    if conf_pct >= 75.0 or (crossover_clean == "BUY_CROSSOVER" and conf_pct >= 60.0):
        return "STRONG BUY 🟢"
    elif conf_pct >= 60.0 or crossover_clean == "BUY_CROSSOVER":
        return "BUY 🟢"
    elif conf_pct <= 35.0 or crossover_clean == "SELL_CROSSOVER":
        return "SELL 🔴"
    else:
        return "HOLD 🟡"


class MLSignalService:
    """
    Broker-independent AI/ML Signal Scoring Engine using Scikit-Learn.
    """

    FEATURE_NAMES = [
        "SMMA 20/120 Distance %",
        "SMMA 20 5-Day Momentum",
        "SMMA 120 5-Day Momentum",
        "Price/SMMA 20 Spread %",
        "Crossover Event Code",
    ]

    def __init__(
        self,
        tech_service: TechnicalIndicatorService | None = None,
        crossover_service: CrossoverService | None = None,
        model_dir: str = "saved_models",
    ) -> None:
        self.tech_service = tech_service or TechnicalIndicatorService()
        self.crossover_service = crossover_service or CrossoverService(tech_service=self.tech_service)
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "quant_signal_rf_v1.joblib"
        self._model: RandomForestClassifier | None = None
        self._initialize_or_load_model()

    def _initialize_or_load_model(self) -> None:
        """Loads serialized model or trains a deterministic baseline RandomForest model."""
        if self.model_path.exists():
            try:
                self._model = joblib.load(self.model_path)
                logger.info(f"Loaded ML model from {self.model_path}")
                return
            except Exception as err:
                logger.warning(f"Could not load existing model from {self.model_path}: {err}")

        # Train a deterministic Random Forest model on historical demo universe
        logger.info("Training deterministic baseline Random Forest ML model...")
        X_train, y_train = self._build_training_dataset()

        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        if not X_train.empty and len(y_train) > 0:
            rf.fit(X_train, y_train)
        else:
            # Fallback dummy fit if empty
            rf.fit([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], [0, 1])

        self._model = rf
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(rf, self.model_path)
            logger.info(f"Saved trained ML model to {self.model_path}")
        except Exception as err:
            logger.warning(f"Could not save trained model to disk: {err}")

    def _build_training_dataset(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Builds training feature matrix from demo universe stock histories."""
        symbols = self.tech_service.provider.get_symbols()
        X_list = []
        y_list = []

        for sym in symbols:
            stock_res = self.tech_service.calculate_stock_indicators(symbol=sym, days=250)
            hist = stock_res.get("history_df", pd.DataFrame())
            if hist.empty or len(hist) < 20:
                continue

            feat_df = extract_features_from_history(hist)
            if feat_df.empty:
                continue

            # Target label: 5-bar forward return positive (1) or negative (0)
            close_series = hist.loc[feat_df.index, "close"]
            future_ret = close_series.shift(-5) - close_series
            target = (future_ret > 0).astype(int).to_numpy()

            # Align lengths
            X_list.append(feat_df.iloc[:-5])
            y_list.append(target[:-5])

        if not X_list:
            return pd.DataFrame(), np.array([])

        X_all = pd.concat(X_list, ignore_index=True)
        y_all = np.concatenate(y_list)
        return X_all, y_all

    def get_feature_importances(self) -> dict[str, float]:
        """Returns feature importance weights calculated by the trained Random Forest model."""
        if self._model is None or not hasattr(self._model, "feature_importances_"):
            return {name: 0.20 for name in self.FEATURE_NAMES}

        importances = self._model.feature_importances_
        return {name: float(imp) for name, imp in zip(self.FEATURE_NAMES, importances)}

    def map_confidence_to_recommendation(self, conf_pct: float, crossover: str = "NONE") -> str:
        """Method wrapper for map_confidence_to_recommendation."""
        return map_confidence_to_recommendation(conf_pct=conf_pct, crossover=crossover)

    def calculate_stock_ml_signal(self, symbol: str) -> dict[str, Any]:
        """
        Calculates AI/ML confidence score and recommendation for a given stock symbol.
        """
        stock_res = self.tech_service.calculate_stock_indicators(symbol=symbol, days=250)
        hist = stock_res.get("history_df", pd.DataFrame())

        if hist.empty or self._model is None:
            return {
                "symbol": symbol,
                "confidence_pct": 50.0,
                "recommendation": "HOLD 🟡",
                "crossover": "NONE",
                "trend": stock_res.get("trend", "Insufficient Data"),
            }

        feat_df = extract_features_from_history(hist)
        if feat_df.empty:
            return {
                "symbol": symbol,
                "confidence_pct": 50.0,
                "recommendation": "HOLD 🟡",
                "crossover": stock_res.get("crossover", "NONE"),
                "trend": stock_res.get("trend", "Insufficient Data"),
            }

        latest_features = feat_df.iloc[[-1]]
        try:
            probs = self._model.predict_proba(latest_features)[0]
            # Bullish win probability
            bull_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            conf_pct = round(bull_prob * 100.0, 2)
        except Exception:
            conf_pct = 50.0

        crossover = stock_res.get("crossover", "NONE")
        rec = map_confidence_to_recommendation(conf_pct=conf_pct, crossover=crossover)

        return {
            "symbol": symbol,
            "ltp": stock_res["ltp"],
            "smma_20": stock_res["smma_20"],
            "smma_120": stock_res["smma_120"],
            "trend": stock_res["trend"],
            "distance_pct": stock_res["distance_pct"],
            "crossover": crossover,
            "confidence_pct": conf_pct,
            "recommendation": rec,
            "latest_features": latest_features.to_dict("records")[0],
        }

    def get_stock_explainability(self, symbol: str) -> dict[str, Any]:
        """
        Step 4D: Generates structured AI decision explainability and feature contribution
        breakdown for an individual stock symbol.
        """
        signal_res = self.calculate_stock_ml_signal(symbol)
        latest_feats = signal_res.get("latest_features", {})
        importances = self.get_feature_importances()

        feat_key_map = [
            ("distance_pct", "SMMA 20/120 Distance %"),
            ("smma_20_slope", "SMMA 20 5-Day Momentum"),
            ("smma_120_slope", "SMMA 120 5-Day Momentum"),
            ("price_smma20_dist", "Price/SMMA 20 Spread %"),
            ("crossover_val", "Crossover Event Code"),
        ]

        feature_rows = []
        for key, name in feat_key_map:
            val = latest_feats.get(key, 0.0)
            imp = importances.get(name, 0.20)
            feature_rows.append({
                "feature": name,
                "value": float(val),
                "importance": float(imp),
            })

        conf = signal_res.get("confidence_pct", 50.0)
        rec = signal_res.get("recommendation", "HOLD 🟡")
        crossover = signal_res.get("crossover", "NONE")

        if "STRONG BUY" in rec:
            explanation = (
                f"STRONG BUY signal generated with high AI confidence of {conf:.1f}%. "
                f"Technical crossover status is '{crossover}', showing strong bullish momentum alignment."
            )
        elif "BUY" in rec:
            explanation = (
                f"BUY signal generated with AI confidence of {conf:.1f}%. "
                f"Technical crossover status is '{crossover}', indicating positive trend direction."
            )
        elif "SELL" in rec:
            explanation = (
                f"SELL signal generated with low AI bullish confidence of {conf:.1f}%. "
                f"Technical crossover status is '{crossover}', indicating downside risk."
            )
        else:
            explanation = (
                f"HOLD signal generated with neutral AI confidence of {conf:.1f}%. "
                f"No strong technical crossover bias detected ('{crossover}')."
            )

        company_name = symbol
        if hasattr(self.tech_service.provider, "_data"):
            comp_info = self.tech_service.provider._data.get(symbol, {})
            company_name = comp_info.get("company", symbol)

        return {
            "symbol": symbol,
            "company_name": company_name,
            "ltp": signal_res.get("ltp", 0.0),
            "trend": signal_res.get("trend", "N/A"),
            "crossover": crossover,
            "confidence_pct": conf,
            "recommendation": rec,
            "explanation": explanation,
            "features": feature_rows,
        }

    def get_ml_signals_dataframe(self, symbols: list[str] | None = None) -> pd.DataFrame:
        """
        Calculates AI/ML signals across all demo universe symbols.
        """
        target_symbols = symbols if symbols is not None else self.tech_service.provider.get_symbols()
        rows = []

        company_map = {}
        if hasattr(self.tech_service.provider, "_data"):
            company_map = {k: v.get("company", k) for k, v in self.tech_service.provider._data.items()}

        for sym in target_symbols:
            ml_res = self.calculate_stock_ml_signal(symbol=sym)
            comp_name = company_map.get(sym, sym)

            crossover_raw = ml_res.get("crossover", "NONE")
            crossover_label = "NO CROSSOVER"
            if crossover_raw == "BUY_CROSSOVER":
                crossover_label = "BUY CROSSOVER"
            elif crossover_raw == "SELL_CROSSOVER":
                crossover_label = "SELL CROSSOVER"

            rows.append({
                "symbol": sym,
                "company_name": comp_name,
                "ltp": ml_res.get("ltp", 0.0),
                "smma_20": ml_res.get("smma_20", np.nan),
                "smma_120": ml_res.get("smma_120", np.nan),
                "trend": ml_res.get("trend", "N/A"),
                "distance_pct": ml_res.get("distance_pct", 0.0),
                "crossover": crossover_label,
                "crossover_raw": crossover_raw,
                "confidence_pct": ml_res.get("confidence_pct", 50.0),
                "recommendation": ml_res.get("recommendation", "HOLD 🟡"),
                "timestamp": "10:30:00 (Simulated Demo Time)",
            })

        if not rows:
            return pd.DataFrame(columns=[
                "symbol", "company_name", "ltp", "smma_20", "smma_120", "trend",
                "distance_pct", "crossover", "crossover_raw", "confidence_pct", "recommendation", "timestamp"
            ])

        df = pd.DataFrame(rows)
        # Sort by AI confidence percentage descending
        return df.sort_values(by="confidence_pct", ascending=False).reset_index(drop=True)

    def get_summary_metrics(self, df_ml: pd.DataFrame | None = None) -> dict[str, Any]:
        """Calculates AI/ML dashboard summary metrics."""
        if df_ml is None:
            df_ml = self.get_ml_signals_dataframe()

        if df_ml.empty:
            return {
                "total_evaluated": 0,
                "high_confidence_count": 0,
                "bullish_recommendations": 0,
                "avg_confidence": 0.0,
            }

        high_conf = len(df_ml[df_ml["confidence_pct"] >= 65.0])
        bullish_recs = len(df_ml[df_ml["recommendation"].str.contains("BUY")])
        avg_conf = float(df_ml["confidence_pct"].mean())

        return {
            "total_evaluated": len(df_ml),
            "high_confidence_count": high_conf,
            "bullish_recommendations": bullish_recs,
            "avg_confidence": avg_conf,
        }
