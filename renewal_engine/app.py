from __future__ import annotations

import argparse

import pandas as pd
import streamlit as st


@st.cache_data
def load_results(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results", default="output.csv")
    args, _ = parser.parse_known_args()

    st.set_page_config(page_title="Renewal Intelligence Engine", layout="wide")
    st.title("Renewal Intelligence Engine")

    df = load_results(args.results)

    risk_filter = st.multiselect("Risk level", sorted(df["risk_level"].dropna().unique()), default=sorted(df["risk_level"].dropna().unique()))
    filtered = df[df["risk_level"].isin(risk_filter)] if risk_filter else df

    st.dataframe(filtered, use_container_width=True)

    st.subheader("Account Details")
    for _, row in filtered.iterrows():
        with st.expander(f"{row.get('account_name', 'Unknown')} ({row.get('risk_level', 'N/A')})"):
            st.markdown(f"**Risk Score:** {row.get('risk_score', 'N/A')}")
            st.markdown(f"**Explanation:** {row.get('explanation', '')}")
            st.markdown(f"**Recommended Action:** {row.get('recommended_action', '')}")


if __name__ == "__main__":
    main()
