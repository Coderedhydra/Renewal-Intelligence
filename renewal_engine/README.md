# Renewal Intelligence Engine

Production-grade prototype for predicting renewal risk using **structured + unstructured** customer signals, and generating human-readable account-level explanations.

## Architecture

```mermaid
flowchart TD
    A[Raw Inputs] --> B[Data Ingestion Layer]
    B --> C[Entity Resolution\n(Account Name Normalization)]
    C --> D[Feature Engineering]
    D --> E[Hybrid Risk Scoring Engine]
    E --> F[LLM Explanation Engine]
    F --> G[Output.csv + CLI]
    G --> H[Optional Streamlit UI]

    A1[accounts.csv] --> A
    A2[usage_metrics.csv] --> A
    A3[support_tickets.csv] --> A
    A4[nps_responses.csv] --> A
    A5[csm_notes.txt] --> A
    A6[changelog.md] --> A
```

## Modules

- `main.py` — CLI entrypoint and pipeline orchestration.
- `data_loader.py` — ingestion, schema checks, parsing `csm_notes.txt` and `changelog.md`.
- `feature_engineering.py` — structured/unstructured feature generation + non-obvious insight flags.
- `risk_model.py` — hybrid rule-based + optional logistic regression scoring.
- `llm_explainer.py` — LLM-backed explanation and action generation with fallback mode.
- `app.py` — optional Streamlit UI.
- `config.py` — central configuration, weights, thresholds.
- `utils.py` — shared utilities (validation, logging, normalization).

## Data Handling Strategy

### 1) Messy Entity Resolution
Account names are normalized (case, punctuation, alias handling) into `account_name_normalized` and merged across all sources.

### 2) Structured + Unstructured Parsing
- `csm_notes.txt` parser extracts:
  - `account_name`
  - sentiment (`positive|neutral|negative`)
  - risk keywords list
- `changelog.md` parser extracts:
  - deprecations
  - breaking changes
  - feature removals
  - optional SDK major version mentions

### 3) Contradictory Signal Handling
A weighted rule engine and optional ML probability are blended:

- Final score = `0.65 * rule_score + 0.35 * ml_score` when ML is available
- Else final score = `rule_score`

This balances explicit business logic and statistical pattern detection.

## Feature Catalog

### Usage Features
- `usage_trend` (`increasing|stable|decreasing`)
- `active_users_ratio`
- `api_usage_growth_pct`
- `usage_drop_pct`

### Support Features
- `ticket_count_last_30_days`
- `p1_ticket_count`
- `unresolved_ticket_age_days`

### NPS Features
- `nps_score`
- `nps_bucket` (`Promoter|Passive|Detractor`)
- `nps_sentiment`

### CSM Notes (LLM-ready) Features
- `churn_risk_flag`
- `competitor_mentions`
- `budget_issues`
- `product_issues`
- `executive_involvement`
- `csm_sentiment`

### Changelog Intelligence
- `deprecated_sdk_usage`
- `breaking_change_exposure`
- `feature_removal_exposure`
- `technical_risk_score`

## Rule Weights

- High P1 tickets (`>2`) → `+20`
- NPS `< 6` → `+15`
- Usage drop `>30%` → `+25`
- Budget mention → `+30`
- Competitor mention → `+25`
- Deprecated SDK usage → `+35`

Additional nudges:
- Stale unresolved tickets (`>21 days`) → `+12`
- Negative exec involvement → `+8`
- Technical risk score contributes scaled additive effect

## Non-Obvious Insights (Differentiators)

1. **Silent Churn Risk**
   - Promoter NPS + declining usage (>20% drop).
   - Interpretation: champion sentiment is positive, but actual product adoption is shrinking.

2. **Relationship Risk**
   - Healthy usage ratio + negative CSM sentiment.
   - Interpretation: operational usage looks fine, but stakeholder trust is degrading.

## LLM Explanation Engine

For each account, the system passes a compact signal bundle to an LLM prompt asking for:
- clear risk explanation
- recommended actions

If API key/client is unavailable, deterministic fallback explanations are generated so pipeline remains production-safe.

## CLI Usage

```bash
python renewal_engine/main.py --input_dir ./data --output output.csv
```

Optional args:
- `--llm_provider openai`
- `--llm_model gpt-4o-mini`
- `--log_level INFO`

## Streamlit UI (Optional)

```bash
streamlit run renewal_engine/app.py -- --results output.csv
```

## Output Schema

Saved CSV contains:
- `account_id`
- `account_name`
- `renewal_date`
- `risk_score`
- `risk_level` (`High|Medium|Low`)
- `explanation`
- `recommended_action`

## Tradeoffs

- Rule-based logic is interpretable but may miss nonlinear interactions.
- Proxy target for ML (without true churn labels) is practical for prototyping but not equivalent to supervised ground truth.
- Regex parsing of notes/changelog is robust for prototype speed, but an NLP parser would improve recall/precision.

## Future Improvements

1. Real-time pipeline (event ingestion + incremental feature updates)
2. Vector DB for semantic CSM notes retrieval and memory
3. Fine-tuned risk model trained on historical renewals/churn
4. Agent-based workflow for account playbook generation and follow-up tracking
5. Confidence intervals and calibration layer for risk scores
