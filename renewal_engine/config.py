from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class RiskWeights:
    high_p1_tickets: int = 20
    low_nps: int = 15
    usage_drop_30_pct: int = 25
    budget_mentions: int = 30
    competitor_mentions: int = 25
    deprecated_sdk_usage: int = 35
    unresolved_ticket_stale: int = 12
    executive_involvement: int = 8


@dataclass(frozen=True)
class RiskThresholds:
    high: int = 70
    medium: int = 40


@dataclass(frozen=True)
class ParsingConfig:
    risk_keywords: List[str] = field(
        default_factory=lambda: [
            "frustrated",
            "churn",
            "budget cut",
            "switching",
            "competitor",
            "escalation",
            "security concern",
            "blocked",
        ]
    )
    competitor_terms: List[str] = field(
        default_factory=lambda: ["salesforce", "hubspot", "zendesk", "freshworks", "oracle"]
    )
    budget_terms: List[str] = field(
        default_factory=lambda: ["budget", "cost", "price", "procurement", "discount"]
    )
    product_issue_terms: List[str] = field(
        default_factory=lambda: ["bug", "outage", "latency", "integration", "incident", "broken"]
    )
    executive_terms: List[str] = field(
        default_factory=lambda: ["vp", "cfo", "cto", "ceo", "executive", "board"]
    )


@dataclass(frozen=True)
class AppConfig:
    input_dir: Path
    output_file: Path
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None
    min_accounts_for_ml: int = 15
    risk_weights: RiskWeights = field(default_factory=RiskWeights)
    risk_thresholds: RiskThresholds = field(default_factory=RiskThresholds)
    parsing: ParsingConfig = field(default_factory=ParsingConfig)

    @classmethod
    def from_runtime(
        cls,
        input_dir: str,
        output_file: str,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
    ) -> "AppConfig":
        return cls(
            input_dir=Path(input_dir),
            output_file=Path(output_file),
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_key=os.getenv("OPENAI_API_KEY"),
        )


ACCOUNT_NAME_ALIASES: Dict[str, str] = {
    "acme inc": "acme",
    "acme corporation": "acme",
    "globex corp": "globex",
    "initech ltd": "initech",
    "soylent co": "soylent",
}


REQUIRED_FILES = [
    "accounts.csv",
    "usage_metrics.csv",
    "support_tickets.csv",
    "nps_responses.csv",
    "csm_notes.txt",
    "changelog.md",
]
