from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config import AppConfig
from utils import normalize_account_name, robust_to_datetime, robust_to_numeric, validate_input_dir

logger = logging.getLogger(__name__)


@dataclass
class LoadedData:
    accounts: pd.DataFrame
    usage: pd.DataFrame
    support: pd.DataFrame
    nps: pd.DataFrame
    csm_notes: pd.DataFrame
    changelog_events: pd.DataFrame


class DataLoader:
    def __init__(self, config: AppConfig):
        self.config = config

    def load_all(self) -> LoadedData:
        validate_input_dir(self.config.input_dir)

        accounts = self._load_accounts(self.config.input_dir / "accounts.csv")
        usage = self._load_usage(self.config.input_dir / "usage_metrics.csv", accounts)
        support = self._load_support(self.config.input_dir / "support_tickets.csv")
        nps = self._load_nps(self.config.input_dir / "nps_responses.csv")
        csm_notes = self._parse_csm_notes(self.config.input_dir / "csm_notes.txt")
        changelog_events = self._parse_changelog(self.config.input_dir / "changelog.md")

        return LoadedData(accounts, usage, support, nps, csm_notes, changelog_events)

    def normalize_schema(
        self,
        df: pd.DataFrame,
        reference_df: pd.DataFrame,
        dataset_name: str,
    ) -> pd.DataFrame:
        """
        Normalizes a source dataframe to guarantee `account_name` when possible.
        If `account_name` is missing and `account_id` exists, enrich from reference_df.
        """
        out = self._canonicalize_columns(df.copy())
        ref = self._canonicalize_columns(reference_df.copy())

        if "account_id" in out.columns:
            out["account_id"] = out["account_id"].astype(str).str.strip()
        if "account_id" in ref.columns:
            ref["account_id"] = ref["account_id"].astype(str).str.strip()

        if "account_name" not in out.columns:
            if "account_id" in out.columns and "account_id" in ref.columns and "account_name" in ref.columns:
                logger.info(
                    "%s missing account_name; backfilling from accounts reference using account_id",
                    dataset_name,
                )
                out = out.merge(
                    ref[["account_id", "account_name"]],
                    on="account_id",
                    how="left",
                )
            else:
                logger.warning(
                    "%s missing account_name and cannot backfill (account_id reference unavailable)",
                    dataset_name,
                )

        if "account_name" in out.columns:
            missing_after = out["account_name"].isna().sum()
            if missing_after > 0:
                logger.warning(
                    "%s has %s rows with missing account_name after schema normalization",
                    dataset_name,
                    int(missing_after),
                )
        return out

    def _canonicalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        aliases = {
            "accountid": "account_id",
            "account_id": "account_id",
            "acct_id": "account_id",
            "customer_id": "account_id",
            "accountname": "account_name",
            "account_name": "account_name",
            "acct_name": "account_name",
            "customer_name": "account_name",
            "createdat": "created_at",
            "updatedat": "updated_at",
            "responsedate": "response_date",
        }
        renamed = {}
        for col in df.columns:
            compact = re.sub(r"[^a-z0-9]", "", col.strip().lower())
            if compact in aliases and aliases[compact] not in df.columns:
                renamed[col] = aliases[compact]
        return df.rename(columns=renamed)

    def _standardize(self, df: pd.DataFrame, account_col: str) -> pd.DataFrame:
        if account_col not in df.columns:
            raise ValueError(f"Expected column '{account_col}' not found in {df.columns.tolist()}")
        out = df.copy()
        out["account_name_normalized"] = out[account_col].map(normalize_account_name)
        return out

    def _load_accounts(self, path: Path) -> pd.DataFrame:
        df = self._canonicalize_columns(pd.read_csv(path))
        for col in ["account_id", "account_name", "renewal_date"]:
            if col not in df.columns:
                raise ValueError(f"accounts.csv missing required column: {col}")
        df["account_id"] = df["account_id"].astype(str).str.strip()
        df = self._standardize(df, "account_name")
        if "arr" in df.columns:
            df["arr"] = robust_to_numeric(df["arr"]).fillna(0.0)
        df["renewal_date"] = robust_to_datetime(df["renewal_date"])
        return df

    def _load_usage(self, path: Path, accounts_df: pd.DataFrame) -> pd.DataFrame:
        df = self._canonicalize_columns(pd.read_csv(path))
        df = self.normalize_schema(df, accounts_df, "usage_metrics.csv")

        required = ["date", "active_users", "api_calls"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"usage_metrics.csv missing required column: {col}")
        if "account_name" not in df.columns:
            raise ValueError(
                "usage_metrics.csv schema normalization failed: expected account_name or account_id->account_name mapping"
            )

        df = self._standardize(df, "account_name")
        df["date"] = robust_to_datetime(df["date"])
        df["active_users"] = robust_to_numeric(df["active_users"]).fillna(0)
        df["api_calls"] = robust_to_numeric(df["api_calls"]).fillna(0)
        if "licensed_users" in df.columns:
            df["licensed_users"] = robust_to_numeric(df["licensed_users"]).replace(0, pd.NA)
        return df

    def _load_support(self, path: Path) -> pd.DataFrame:
        df = self._canonicalize_columns(pd.read_csv(path))
        required = ["account_name", "created_at", "severity", "status"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"support_tickets.csv missing required column: {col}")
        df = self._standardize(df, "account_name")
        df["created_at"] = robust_to_datetime(df["created_at"])
        if "updated_at" in df.columns:
            df["updated_at"] = robust_to_datetime(df["updated_at"])
        return df

    def _load_nps(self, path: Path) -> pd.DataFrame:
        df = self._canonicalize_columns(pd.read_csv(path))
        required = ["account_name", "score", "comment", "response_date"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"nps_responses.csv missing required column: {col}")
        df = self._standardize(df, "account_name")
        df["score"] = robust_to_numeric(df["score"]).clip(lower=0, upper=10)
        df["response_date"] = robust_to_datetime(df["response_date"])
        return df

    def _parse_csm_notes(self, path: Path) -> pd.DataFrame:
        text = path.read_text(encoding="utf-8")
        blocks = [blk.strip() for blk in re.split(r"\n\s*\n", text) if blk.strip()]
        rows: List[Dict[str, object]] = []
        for block in blocks:
            acct, note = self._extract_account_and_body(block)
            if not acct or not note:
                continue
            sentiment = self._simple_sentiment(note)
            keywords = [kw for kw in self.config.parsing.risk_keywords if kw in note.lower()]
            rows.append(
                {
                    "account_name": acct,
                    "account_name_normalized": normalize_account_name(acct),
                    "note_text": note,
                    "sentiment": sentiment,
                    "risk_keywords": keywords,
                }
            )

        csm_df = pd.DataFrame(rows)
        logger.info("Parsed %s CSM note records", len(csm_df))
        return csm_df

    def _extract_account_and_body(self, block: str) -> Tuple[str, str]:
        # Supports formats like "Account: Acme" or leading "[Acme]"
        account_match = re.search(r"(?:account\s*:\s*)(.+)", block, flags=re.IGNORECASE)
        if account_match:
            account = account_match.group(1).splitlines()[0].strip()
            note = block.replace(account_match.group(0), "", 1).strip()
            return account, note

        bracket_match = re.match(r"\[(.+?)\]\s*(.*)", block, flags=re.DOTALL)
        if bracket_match:
            return bracket_match.group(1).strip(), bracket_match.group(2).strip()

        # fallback "Acme - note ..."
        dash = block.split("-", 1)
        if len(dash) == 2 and len(dash[0].split()) <= 5:
            return dash[0].strip(), dash[1].strip()
        return "", ""

    def _simple_sentiment(self, note: str) -> str:
        low = note.lower()
        negative_terms = ["frustrated", "angry", "blocked", "churn", "issue", "unhappy"]
        positive_terms = ["happy", "great", "expanding", "renew", "successful"]
        neg = sum(t in low for t in negative_terms)
        pos = sum(t in low for t in positive_terms)
        if neg > pos:
            return "negative"
        if pos > neg:
            return "positive"
        return "neutral"

    def _parse_changelog(self, path: Path) -> pd.DataFrame:
        text = path.read_text(encoding="utf-8")
        rows = []
        current_date = None
        for line in text.splitlines():
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", line)
            if date_match:
                current_date = pd.to_datetime(date_match.group(1), errors="coerce", utc=True)

            low = line.lower()
            event_type = None
            if "breaking" in low:
                event_type = "breaking_change"
            elif "deprecat" in low:
                event_type = "deprecation"
            elif "remov" in low:
                event_type = "feature_removal"

            if event_type:
                sdk_match = re.search(r"sdk\s*v?(\d+)", low)
                sdk_major = int(sdk_match.group(1)) if sdk_match else None
                rows.append(
                    {
                        "event_date": current_date,
                        "event_type": event_type,
                        "description": line.strip(),
                        "sdk_major": sdk_major,
                    }
                )

        changelog_df = pd.DataFrame(rows)
        logger.info("Parsed %s changelog risk events", len(changelog_df))
        return changelog_df
