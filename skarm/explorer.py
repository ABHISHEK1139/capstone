"""Interactive Rule Explorer and reporting utility."""

from typing import Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np


class RuleExplorer:
    """Explores, filters, and formats mined association rules."""

    def __init__(self, rules_df: pd.DataFrame):
        self.rules_df = rules_df.copy()

    def filter_rules(
        self,
        min_confidence: Optional[float] = None,
        min_lift: Optional[float] = None,
        min_support: Optional[float] = None,
        antecedent_contains: Optional[Union[str, List[str]]] = None,
        consequent_contains: Optional[Union[str, List[str]]] = None,
        sort_by: Optional[str] = None,
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Filter rules by thresholds and items, with optional sorting."""
        filtered = self.rules_df.copy()

        if min_confidence is not None:
            filtered = filtered[filtered["confidence"] >= min_confidence]
        if min_lift is not None:
            filtered = filtered[filtered["lift"] >= min_lift]
        if min_support is not None:
            filtered = filtered[filtered["support"] >= min_support]

        if antecedent_contains is not None:
            items = [antecedent_contains] if isinstance(antecedent_contains, str) else antecedent_contains
            col = "antecedent_names" if "antecedent_names" in filtered.columns else "antecedent"
            filtered = filtered[filtered[col].apply(lambda s: any(item in s for item in items))]

        if consequent_contains is not None:
            items = [consequent_contains] if isinstance(consequent_contains, str) else consequent_contains
            col = "consequent_names" if "consequent_names" in filtered.columns else "consequent"
            filtered = filtered[filtered[col].apply(lambda s: any(item in s for item in items))]

        if sort_by is not None and sort_by in filtered.columns:
            filtered = filtered.sort_values(by=sort_by, ascending=ascending)

        return filtered.reset_index(drop=True)

    def summary_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive rule summary statistics."""
        if self.rules_df.empty:
            return {
                "rule_count": 0,
                "avg_confidence": 0.0,
                "avg_lift": 0.0,
                "avg_support": 0.0,
                "max_lift": 0.0,
            }

        return {
            "rule_count": len(self.rules_df),
            "avg_confidence": float(self.rules_df["confidence"].mean()),
            "avg_lift": float(self.rules_df["lift"].mean()),
            "avg_support": float(self.rules_df["support"].mean()),
            "max_lift": float(self.rules_df["lift"].max()),
            "min_lift": float(self.rules_df["lift"].min()),
            "avg_leverage": float(self.rules_df["leverage"].mean()),
            "avg_zhangs_metric": float(self.rules_df["zhangs_metric"].mean()),
        }

    def to_markdown(self, df: Optional[Union[pd.DataFrame, int]] = None, top_n: int = 10) -> str:
        """Render top N rules or provided DataFrame as a formatted markdown table without external dependencies."""
        if isinstance(df, pd.DataFrame):
            target_df = df
            n = top_n
        elif isinstance(df, int):
            target_df = self.rules_df
            n = df
        else:
            target_df = self.rules_df
            n = top_n

        if target_df.empty:
            return "*No rules generated.*"

        display_df = target_df.head(n).copy()
        col_ante = "antecedent_names" if "antecedent_names" in display_df.columns else "antecedent"
        col_cons = "consequent_names" if "consequent_names" in display_df.columns else "consequent"

        display_df["Rule"] = display_df.apply(
            lambda r: f"{sorted(list(r[col_ante]))} -> {sorted(list(r[col_cons]))}",
            axis=1,
        )

        cols = ["Rule", "support", "confidence", "lift", "leverage", "conviction", "zhangs_metric"]
        sub = display_df[cols].round(4)

        # Build markdown table manually to avoid tabulate dependency
        headers = [str(c) for c in cols]
        header_row = "| " + " | ".join(headers) + " |"
        sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        data_rows = []
        for _, row in sub.iterrows():
            row_str = "| " + " | ".join(str(row[c]) for c in cols) + " |"
            data_rows.append(row_str)

        return "\n".join([header_row, sep_row] + data_rows)
