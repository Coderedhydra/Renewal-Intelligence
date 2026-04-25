from __future__ import annotations

import json
import logging
from typing import Tuple

import pandas as pd

from config import AppConfig

logger = logging.getLogger(__name__)


class LLMExplainer:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = None
        if config.llm_provider.lower() == "openai" and config.llm_api_key:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=config.llm_api_key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("OpenAI client unavailable, fallback mode enabled: %s", exc)

    def explain(self, row: pd.Series) -> Tuple[str, str]:
        signals = {
            "usage_trend": row.get("usage_trend"),
            "usage_drop_pct": row.get("usage_drop_pct"),
            "active_users_ratio": row.get("active_users_ratio"),
            "support_issues": {
                "ticket_count_last_30_days": row.get("ticket_count_last_30_days"),
                "p1_ticket_count": row.get("p1_ticket_count"),
                "unresolved_ticket_age_days": row.get("unresolved_ticket_age_days"),
            },
            "nps": {
                "score": row.get("nps_score"),
                "bucket": row.get("nps_bucket"),
                "sentiment": row.get("nps_sentiment"),
            },
            "csm_notes_summary": row.get("notes_summary", "No CSM notes available"),
            "csm_flags": {
                "churn_risk_flag": row.get("churn_risk_flag"),
                "competitor_mentions": row.get("competitor_mentions"),
                "budget_issues": row.get("budget_issues"),
                "product_issues": row.get("product_issues"),
                "executive_involvement": row.get("executive_involvement"),
            },
            "technical_risks": {
                "technical_risk_score": row.get("technical_risk_score"),
                "deprecated_sdk_usage": row.get("deprecated_sdk_usage"),
                "breaking_change_exposure": row.get("breaking_change_exposure"),
                "feature_removal_exposure": row.get("feature_removal_exposure"),
            },
            "non_obvious": {
                "silent_churn_risk": row.get("silent_churn_risk"),
                "relationship_risk": row.get("relationship_risk"),
            },
            "risk_level": row.get("risk_level"),
            "risk_score": row.get("risk_score"),
        }

        if self.client:
            return self._generate_with_openai(signals)
        return self._generate_fallback(signals)

    def _generate_with_openai(self, signals: dict) -> Tuple[str, str]:
        prompt = (
            "Explain why this account is at renewal risk based on the following signals. "
            "Be specific and causal. Then provide concise recommended actions. "
            "Return strict JSON with keys: explanation, recommended_action.\n\n"
            f"Signals:\n{json.dumps(signals, indent=2, default=str)}"
        )

        try:
            response = self.client.responses.create(
                model=self.config.llm_model,
                input=prompt,
                temperature=0.2,
            )
            text = response.output_text
            parsed = json.loads(text)
            return parsed["explanation"], parsed["recommended_action"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM explanation failed, using fallback template: %s", exc)
            return self._generate_fallback(signals)

    def _generate_fallback(self, signals: dict) -> Tuple[str, str]:
        risk_factors = []
        actions = []

        if signals["usage_trend"] == "decreasing":
            risk_factors.append("Product usage is declining, indicating weakening adoption")
            actions.append("Run adoption recovery plan with weekly value reviews")
        if signals["support_issues"]["p1_ticket_count"] and signals["support_issues"]["p1_ticket_count"] > 2:
            risk_factors.append("High volume of critical support incidents erodes confidence")
            actions.append("Create executive escalation path and 14-day incident remediation")
        if signals["nps"]["score"] is not None and float(signals["nps"]["score"]) < 6:
            risk_factors.append("Detractor NPS suggests dissatisfied stakeholders")
            actions.append("Schedule VOC session and commit to measurable improvements")
        if signals["csm_flags"]["budget_issues"]:
            risk_factors.append("Budget pressure may block renewal unless ROI is proven")
            actions.append("Provide ROI business case and right-sized commercial proposal")
        if signals["csm_flags"]["competitor_mentions"]:
            risk_factors.append("Competitive alternatives are being evaluated")
            actions.append("Launch competitive defense with migration risk analysis")
        if signals["technical_risks"]["deprecated_sdk_usage"]:
            risk_factors.append("Deprecated SDK exposure introduces technical and timeline risk")
            actions.append("Offer migration support plan before deprecation deadline")
        if signals["non_obvious"]["silent_churn_risk"]:
            risk_factors.append("Silent churn risk: positive sentiment but shrinking usage")
            actions.append("Investigate champion sentiment vs real end-user behavior gap")
        if signals["non_obvious"]["relationship_risk"]:
            risk_factors.append("Relationship risk: healthy usage but negative stakeholder tone")
            actions.append("Rebuild executive trust with QBR and issue ownership")

        if not risk_factors:
            risk_factors.append("No acute red flags; monitor for early warning shifts")
            actions.append("Maintain cadence and validate expansion opportunities")

        explanation = "; ".join(risk_factors) + "."
        recommended_action = "; ".join(actions[:3]) + "."
        return explanation, recommended_action
