"""Dataset loader utilities for benchmark and real-world transactional datasets."""
from __future__ import annotations

import os
from typing import List, Tuple, Dict, Any

def load_groceries(filepath: str = "data/groceries.csv") -> List[List[str]]:
    """Load the Groceries dataset (CSV format with comma-separated item names)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Groceries dataset not found at {filepath}")
    
    transactions: List[List[str]] = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                items = [item.strip() for item in line.split(",") if item.strip()]
                if items:
                    transactions.append(items)
    return transactions

def load_fimi_dat(filepath: str) -> List[List[str]]:
    """Load space-separated transaction files in FIMI .dat format (e.g. chess.dat, retail.dat)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    transactions: List[List[str]] = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                items = line.split()
                if items:
                    transactions.append(items)
    return transactions

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
    """Compute structural metrics of a transactional dataset."""
    num_tx = len(transactions)
    unique_items = set()
    total_len = 0
    min_len = float("inf") if num_tx > 0 else 0
    max_len = 0

    for tx in transactions:
        tx_len = len(tx)
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
