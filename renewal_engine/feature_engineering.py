from __future__ import annotations

import logging
import re
from datetime import timedelta

import numpy as np
import pandas as pd

from config import AppConfig
from data_loader import LoadedData

logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, config: AppConfig):
        self.config = config

    def build_feature_table(self, loaded: LoadedData) -> pd.DataFrame:
        base = loaded.accounts.copy()

        usage_features = self._usage_features(loaded.usage)
        support_features = self._support_features(loaded.support)
        nps_features = self._nps_features(loaded.nps)
        csm_features = self._csm_features(loaded.csm_notes)
        tech_risk = self._technical_risk_features(loaded.accounts, loaded.changelog_events)

        frames = [usage_features, support_features, nps_features, csm_features, tech_risk]
        for feat in frames:
            base = base.merge(feat, how="left", on="account_name_normalized")

        defaults = {
            "usage_trend": "stable",
            "active_users_ratio": 0.0,
            "api_usage_growth_pct": 0.0,
            "ticket_count_last_30_days": 0,
            "p1_ticket_count": 0,
            "unresolved_ticket_age_days": 0,
            "nps_score": 7,
            "nps_bucket": "Passive",
            "nps_sentiment": "neutral",
            "churn_risk_flag": 0,
            "competitor_mentions": 0,
            "budget_issues": 0,
            "product_issues": 0,
            "executive_involvement": 0,
            "csm_sentiment": "neutral",
            "technical_risk_score": 0,
            "deprecated_sdk_usage": 0,
            "breaking_change_exposure": 0,
            "feature_removal_exposure": 0,
            "silent_churn_risk": 0,
            "relationship_risk": 0,
        }
        for col, value in defaults.items():
            if col in base.columns:
                base[col] = base[col].fillna(value)

        base["renewal_date"] = pd.to_datetime(base["renewal_date"], errors="coerce", utc=True)
        return base

    def _usage_features(self, usage: pd.DataFrame) -> pd.DataFrame:
        rows = []
        usage = usage.sort_values(["account_name_normalized", "date"])
        for acct, group in usage.groupby("account_name_normalized"):
            recent = group.tail(6)
            if recent.empty:
                continue
            first = recent.iloc[0]
            last = recent.iloc[-1]
            active_first = max(float(first["active_users"]), 1.0)
            api_first = max(float(first["api_calls"]), 1.0)
            active_growth = (float(last["active_users"]) - active_first) / active_first
            api_growth = (float(last["api_calls"]) - api_first) / api_first

            if active_growth > 0.1:
                trend = "increasing"
            elif active_growth < -0.1:
                trend = "decreasing"
            else:
                trend = "stable"

            if "licensed_users" in recent.columns and recent["licensed_users"].notna().any():
                ratio = float(last["active_users"]) / max(float(recent["licensed_users"].dropna().iloc[-1]), 1.0)
            else:
                ratio = float(last["active_users"]) / max(float(recent["active_users"].max()), 1.0)

            rows.append(
                {
                    "account_name_normalized": acct,
                    "usage_trend": trend,
                    "usage_drop_pct": max(-active_growth, 0.0),
                    "active_users_ratio": round(ratio, 4),
                    "api_usage_growth_pct": round(api_growth * 100, 2),
                }
            )

        return pd.DataFrame(rows)

    def _support_features(self, support: pd.DataFrame) -> pd.DataFrame:
        now = pd.Timestamp.utcnow()
        cutoff = now - timedelta(days=30)
        rows = []

        for acct, group in support.groupby("account_name_normalized"):
            recent = group[group["created_at"] >= cutoff]
            unresolved = group[group["status"].str.lower().isin(["open", "pending", "new"])]
            if unresolved.empty:
                unresolved_age = 0
            else:
                unresolved_age = int((now - unresolved["created_at"].min()).days)

            p1_count = int(group["severity"].astype(str).str.upper().isin(["P1", "SEV1", "CRITICAL"]).sum())
            rows.append(
                {
                    "account_name_normalized": acct,
                    "ticket_count_last_30_days": int(len(recent)),
                    "p1_ticket_count": p1_count,
                    "unresolved_ticket_age_days": unresolved_age,
                }
            )
        return pd.DataFrame(rows)

    def _nps_features(self, nps: pd.DataFrame) -> pd.DataFrame:
        nps = nps.sort_values(["account_name_normalized", "response_date"])
        latest = nps.groupby("account_name_normalized", as_index=False).tail(1)

        def bucket(score: float) -> str:
            if score >= 9:
                return "Promoter"
            if score >= 7:
                return "Passive"
            return "Detractor"

        latest["nps_bucket"] = latest["score"].map(bucket)
        latest["nps_sentiment"] = latest["comment"].fillna("").str.lower().map(self._simple_comment_sentiment)
        return latest[["account_name_normalized", "score", "nps_bucket", "nps_sentiment"]].rename(
            columns={"score": "nps_score"}
        )

    def _csm_features(self, csm_notes: pd.DataFrame) -> pd.DataFrame:
        if csm_notes.empty:
            return pd.DataFrame(columns=["account_name_normalized"])

        parsing = self.config.parsing

        def has_any(text: str, terms: list[str]) -> int:
            low = str(text).lower()
            return int(any(t in low for t in terms))

        csm_notes = csm_notes.copy()
        csm_notes["churn_risk_flag"] = csm_notes["note_text"].str.lower().str.contains("churn|renewal risk|at risk").astype(int)
        csm_notes["competitor_mentions"] = csm_notes["note_text"].map(lambda t: has_any(t, parsing.competitor_terms))
        csm_notes["budget_issues"] = csm_notes["note_text"].map(lambda t: has_any(t, parsing.budget_terms))
        csm_notes["product_issues"] = csm_notes["note_text"].map(lambda t: has_any(t, parsing.product_issue_terms))
        csm_notes["executive_involvement"] = csm_notes["note_text"].map(lambda t: has_any(t, parsing.executive_terms))

        sentiment_rank = {"negative": -1, "neutral": 0, "positive": 1}
        csm_notes["sentiment_rank"] = csm_notes["sentiment"].map(sentiment_rank).fillna(0)

        agg = (
            csm_notes.groupby("account_name_normalized", as_index=False)
            .agg(
                churn_risk_flag=("churn_risk_flag", "max"),
                competitor_mentions=("competitor_mentions", "sum"),
                budget_issues=("budget_issues", "sum"),
                product_issues=("product_issues", "sum"),
                executive_involvement=("executive_involvement", "max"),
                csm_sentiment_rank=("sentiment_rank", "mean"),
                notes_summary=("note_text", lambda s: " | ".join(s.tail(3))),
            )
            .reset_index(drop=True)
        )

        agg["csm_sentiment"] = agg["csm_sentiment_rank"].map(lambda x: "negative" if x < -0.2 else "positive" if x > 0.2 else "neutral")
        return agg.drop(columns=["csm_sentiment_rank"])

    def _technical_risk_features(self, accounts: pd.DataFrame, changelog_events: pd.DataFrame) -> pd.DataFrame:
        rows = []
        deprecations = changelog_events[changelog_events["event_type"] == "deprecation"]
        breakings = changelog_events[changelog_events["event_type"] == "breaking_change"]
        removals = changelog_events[changelog_events["event_type"] == "feature_removal"]

        for _, row in accounts.iterrows():
            acct = row["account_name_normalized"]
            tech_stack = str(row.get("product_version", "")) + " " + str(row.get("sdk_version", ""))
            sdk_numbers = [int(n) for n in re.findall(r"v?(\d+)", tech_stack.lower())]
            account_sdk = sdk_numbers[0] if sdk_numbers else None

            deprecated_hit = 0
            if account_sdk is not None and not deprecations.empty:
                deprecated_hit = int((deprecations["sdk_major"].dropna() >= account_sdk).any())

            breaking_exposure = int(not breakings.empty)
            feature_removal_exposure = int(not removals.empty)

            technical_risk_score = 0
            if deprecated_hit:
                technical_risk_score += 35
            technical_risk_score += 10 * breaking_exposure
            technical_risk_score += 8 * feature_removal_exposure

            rows.append(
                {
                    "account_name_normalized": acct,
                    "deprecated_sdk_usage": deprecated_hit,
                    "breaking_change_exposure": breaking_exposure,
                    "feature_removal_exposure": feature_removal_exposure,
                    "technical_risk_score": technical_risk_score,
                }
            )

        return pd.DataFrame(rows)

    def enrich_with_non_obvious_insights(self, features: pd.DataFrame) -> pd.DataFrame:
        out = features.copy()
        out["silent_churn_risk"] = (
            (out["nps_bucket"] == "Promoter")
            & (out["usage_trend"] == "decreasing")
            & (out["usage_drop_pct"] > 0.2)
        ).astype(int)

        out["relationship_risk"] = (
            (out["active_users_ratio"] > 0.7)
            & (out["csm_sentiment"] == "negative")
        ).astype(int)

        return out

    @staticmethod
    def _simple_comment_sentiment(text: str) -> str:
        low = str(text).lower()
        positive = sum(word in low for word in ["love", "great", "excellent", "helpful"])
        negative = sum(word in low for word in ["bad", "poor", "frustrated", "slow", "issue"])
        if negative > positive:
            return "negative"
        if positive > negative:
            return "positive"
        return "neutral"
