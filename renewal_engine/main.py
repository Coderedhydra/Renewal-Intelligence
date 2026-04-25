from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from config import AppConfig
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from llm_explainer import LLMExplainer
from risk_model import RiskScoringEngine
from utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Renewal Intelligence Engine")
    parser.add_argument("--input_dir", required=True, help="Directory containing source datasets")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--llm_provider", default="openai", help="LLM provider (default: openai)")
    parser.add_argument("--llm_model", default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--log_level", default="INFO", help="Logging level")
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    config = AppConfig.from_runtime(
        input_dir=args.input_dir,
        output_file=args.output,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )

    logger.info("Loading data from %s", config.input_dir)
    loader = DataLoader(config)
    loaded = loader.load_all()

    logger.info("Engineering features")
    engineer = FeatureEngineer(config)
    features = engineer.build_feature_table(loaded)
    features = engineer.enrich_with_non_obvious_insights(features)

    logger.info("Scoring risk")
    scorer = RiskScoringEngine(config)
    scored = scorer.score(features)

    logger.info("Generating account explanations")
    explainer = LLMExplainer(config)
    explanations = scored.apply(lambda row: explainer.explain(row), axis=1)
    scored["explanation"] = explanations.map(lambda x: x[0])
    scored["recommended_action"] = explanations.map(lambda x: x[1])

    output_cols = [
        "account_id",
        "account_name",
        "renewal_date",
        "risk_score",
        "risk_level",
        "explanation",
        "recommended_action",
    ]
    existing_cols = [c for c in output_cols if c in scored.columns]
    output_df = scored[existing_cols].copy()
    output_df = output_df.sort_values("risk_score", ascending=False)

    output_df.to_csv(config.output_file, index=False)
    logger.info("Saved output to %s", config.output_file)
    return output_df


def main() -> int:
    args = parse_args()
    try:
        output = run_pipeline(args)
        print(output.head(20).to_string(index=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
