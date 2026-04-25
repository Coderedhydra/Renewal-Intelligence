from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import AppConfig

logger = logging.getLogger(__name__)


class RiskScoringEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        self.model: Pipeline | None = None

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        scored = df.copy()
        scored["rule_risk_score"] = scored.apply(self._rule_score_row, axis=1)

        ml_scores = self._ml_scores(scored)
        if ml_scores is not None:
            scored["ml_risk_score"] = ml_scores
            scored["risk_score"] = (0.65 * scored["rule_risk_score"] + 0.35 * scored["ml_risk_score"]).round(2)
        else:
            scored["ml_risk_score"] = np.nan
            scored["risk_score"] = scored["rule_risk_score"].round(2)

        scored["risk_level"] = scored["risk_score"].map(self._risk_level)
        return scored

    def _rule_score_row(self, row: pd.Series) -> float:
        w = self.config.risk_weights
        score = 0.0
        if row.get("p1_ticket_count", 0) > 2:
            score += w.high_p1_tickets
        if row.get("nps_score", 7) < 6:
            score += w.low_nps
        if row.get("usage_drop_pct", 0) > 0.30:
            score += w.usage_drop_30_pct
        if row.get("budget_issues", 0) > 0:
            score += w.budget_mentions
        if row.get("competitor_mentions", 0) > 0:
            score += w.competitor_mentions
        if row.get("deprecated_sdk_usage", 0) > 0:
            score += w.deprecated_sdk_usage
        if row.get("unresolved_ticket_age_days", 0) > 21:
            score += w.unresolved_ticket_stale
        if row.get("executive_involvement", 0) > 0 and row.get("csm_sentiment", "neutral") == "negative":
            score += w.executive_involvement

        score += float(row.get("technical_risk_score", 0)) * 0.35

        # non-obvious insights as additive nudges
        if row.get("silent_churn_risk", 0):
            score += 10
        if row.get("relationship_risk", 0):
            score += 8

        return min(score, 100.0)

    def _ml_scores(self, df: pd.DataFrame) -> np.ndarray | None:
        if len(df) < self.config.min_accounts_for_ml:
            logger.info("Skipping ML layer: only %s accounts, need at least %s", len(df), self.config.min_accounts_for_ml)
            return None

        train_df = df.copy()
        train_df["target_proxy"] = (
            (train_df["rule_risk_score"] >= 55)
            | (train_df["churn_risk_flag"] > 0)
            | (train_df["nps_score"] <= 5)
        ).astype(int)

        categorical = ["usage_trend", "nps_bucket", "nps_sentiment", "csm_sentiment"]
        numeric = [
            "active_users_ratio",
            "api_usage_growth_pct",
            "ticket_count_last_30_days",
            "p1_ticket_count",
            "unresolved_ticket_age_days",
            "nps_score",
            "competitor_mentions",
            "budget_issues",
            "product_issues",
            "executive_involvement",
            "technical_risk_score",
            "usage_drop_pct",
            "silent_churn_risk",
            "relationship_risk",
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical,
                ),
            ]
        )

        self.model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )

        self.model.fit(train_df[categorical + numeric], train_df["target_proxy"])
        probs = self.model.predict_proba(train_df[categorical + numeric])[:, 1]
        return probs * 100

    def _risk_level(self, score: float) -> str:
        thr = self.config.risk_thresholds
        if score >= thr.high:
            return "High"
        if score >= thr.medium:
            return "Medium"
        return "Low"
