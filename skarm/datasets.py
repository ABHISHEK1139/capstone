"""Dataset loader utilities for benchmark and real-world transactional datasets."""
from __future__ import annotations

import os
from typing import List, Dict, Any

# SPMF sequence-format markers. FIMI item identifiers are non-negative, so
# these two tokens can never be genuine items and are safe to drop.
_SPMF_TOKENS = ("-1", "-2")


def _strip_spmf(tokens: List[str]) -> List[str]:
    """Drop SPMF item separators (-1) and transaction terminators (-2)."""
    return [tok for tok in tokens if tok not in _SPMF_TOKENS]


def _drop_fimi_header(lines: List[str]) -> List[str]:
    """Drop a leading FIMI-style header line, if one is present.

    Some FIMI mirrors (e.g. accidents.dat) start with a single line
    ``N M`` giving the transaction count N and the item count M. Treating
    that line as a transaction would invent a bogus transaction and a bogus
    item. The line is only treated as a header when BOTH numbers check out
    against the rest of the file (N == remaining line count and M == distinct
    item count); a genuine two-item first transaction essentially never
    satisfies both conditions at once.
    """
    if not lines:
        return lines
    head = lines[0].split()
    if len(head) != 2 or not head[0].isdigit() or not head[1].isdigit():
        return lines
    n_expected, m_expected = int(head[0]), int(head[1])
    rest = lines[1:]
    if n_expected != len(rest):
        return lines
    distinct = set()
    for line in rest:
        distinct.update(_strip_spmf(line.split()))
    if m_expected != len(distinct):
        return lines
    return rest


def _read_nonempty_lines(filepath: str) -> List[str]:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f if line.strip()]


def load_groceries(filepath: str = "data/groceries.csv") -> List[List[str]]:
    """Load a delimiter-separated transaction file (CSV or TSV).

    The delimiter is sniffed from the first lines: tab-separated files
    (e.g. BMS-WebView-1) use tabs, everything else uses commas. Splitting a
    tab-separated file on commas would silently merge each transaction into
    a single bogus item, so sniffing is not optional.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Groceries dataset not found at {filepath}")

    lines = _read_nonempty_lines(filepath)
    delimiter = "\t" if any("\t" in line for line in lines[:10]) else ","
    transactions: List[List[str]] = []
    for line in lines:
        items = [item.strip() for item in line.split(delimiter) if item.strip()]
        if items:
            transactions.append(items)
    return transactions


def load_fimi_dat(filepath: str) -> List[List[str]]:
    """Load space-separated transaction files in FIMI .dat format (e.g. chess.dat, retail.dat).

    Handles a leading ``N M`` header line (accidents.dat) and SPMF
    sequence markers (``-1`` separators, ``-2`` terminators) that would
    otherwise be mistaken for items.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    lines = _drop_fimi_header(_read_nonempty_lines(filepath))
    transactions: List[List[str]] = []
    for line in lines:
        items = _strip_spmf(line.split())
        if items:
            transactions.append(items)
    return transactions


def load_transactions(filepath: str) -> List[List[str]]:
    """Load a transaction file, auto-detecting its format.

    Delimiter-separated files (comma or tab) go through the CSV/TSV path;
    whitespace-separated files go through the FIMI path (header detection
    plus SPMF marker stripping). Raises FileNotFoundError for a missing file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    lines = _read_nonempty_lines(filepath)
    if not lines:
        return []
    if any("," in line or "\t" in line for line in lines[:10]):
        return load_groceries(filepath)
    return load_fimi_dat(filepath)


def load_chess(filepath: str = "data/chess.dat") -> List[List[str]]:
    """Load the Chess dataset (dense benchmark)."""
    return load_fimi_dat(filepath)


def load_retail(filepath: str = "data/retail.dat") -> List[List[str]]:
    """Load the Retail dataset (sparse benchmark, 88,162 transactions)."""
    return load_fimi_dat(filepath)


def load_market_basket(filepath: str = "data/market_basket.csv") -> List[List[str]]:
    """Load the Market Basket Optimization dataset (7,501 retail transactions)."""
    return load_groceries(filepath)


def load_mushroom(filepath: str = "data/mushroom.dat") -> List[List[str]]:
    """Load the Mushroom dataset (dense benchmark, 8,124 transactions, 119 attributes)."""
    return load_fimi_dat(filepath)


def dataset_summary(transactions: List[List[str]]) -> Dict[str, Any]:
    """Compute structural metrics of a transactional dataset.

    Transaction length counts distinct items: a transaction is a set of
    items, so duplicated tokens within one line must not inflate the
    average length or the density.
    """
    num_tx = len(transactions)
    unique_items = set()
    total_len = 0
    min_len = float("inf") if num_tx > 0 else 0
    max_len = 0

    for tx in transactions:
        tx_len = len(set(tx))
        total_len += tx_len
        min_len = min(min_len, tx_len)
        max_len = max(max_len, tx_len)
        unique_items.update(tx)

    num_items = len(unique_items)
    avg_len = total_len / num_tx if num_tx > 0 else 0.0
    density = (total_len / (num_tx * num_items)) * 100.0 if (num_tx > 0 and num_items > 0) else 0.0
    sparsity = 100.0 - density

    return {
        "n_transactions": num_tx,
        "n_unique_items": num_items,
        "avg_transaction_length": round(avg_len, 2),
        "min_transaction_length": int(min_len) if min_len != float("inf") else 0,
        "max_transaction_length": int(max_len),
        "density_pct": round(density, 2),
        "sparsity_pct": round(sparsity, 2),
    }
