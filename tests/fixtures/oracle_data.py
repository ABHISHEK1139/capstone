"""Hand-computed fixtures and analytical oracles for CP1 validation."""

from typing import List, Dict, Tuple, FrozenSet, Any


HAND_COMPUTED_TRANSACTIONS = [
    ["A", "B", "C"],
    ["A", "C"],
    ["A", "D"],
    ["B", "E", "F"],
    ["A", "B", "C", "E"],
]

# Oracle frequent itemsets for min_support = 0.4 (count >= 2 out of 5)
ORACLE_FREQUENT_ITEMSETS_0_4 = {
    frozenset(["A"]): 0.8,
    frozenset(["B"]): 0.6,
    frozenset(["C"]): 0.6,
    frozenset(["E"]): 0.4,
    frozenset(["A", "B"]): 0.4,
    frozenset(["A", "C"]): 0.6,
    frozenset(["B", "C"]): 0.4,
    frozenset(["B", "E"]): 0.4,
    frozenset(["A", "B", "C"]): 0.4,
}

# Oracle rules for min_confidence = 0.6 on min_support = 0.4
# (antecedent, consequent): (supp_a, supp_c, supp_ac, conf, lift, leverage, conviction)
ORACLE_RULES_0_4_CONF_0_6: Dict[Tuple[FrozenSet[str], FrozenSet[str]], Dict[str, float]] = {
    (frozenset(["A"]), frozenset(["C"])): {
        "antecedent_support": 0.8,
        "consequent_support": 0.6,
        "support": 0.6,
        "confidence": 0.75,
        "lift": 1.25,
        "leverage": 0.12,
        "conviction": 1.6,
    },
    (frozenset(["C"]), frozenset(["A"])): {
        "antecedent_support": 0.6,
        "consequent_support": 0.8,
        "support": 0.6,
        "confidence": 1.0,
        "lift": 1.25,
        "leverage": 0.12,
        "conviction": float("inf"),
    },
    (frozenset(["B"]), frozenset(["A"])): {
        "antecedent_support": 0.6,
        "consequent_support": 0.8,
        "support": 0.4,
        "confidence": 2.0 / 3.0,
        "lift": (2.0 / 3.0) / 0.8,
        "leverage": 0.4 - (0.6 * 0.8),
        "conviction": (1.0 - 0.8) / (1.0 - 2.0 / 3.0),
    },
    (frozenset(["B"]), frozenset(["C"])): {
        "antecedent_support": 0.6,
        "consequent_support": 0.6,
        "support": 0.4,
        "confidence": 2.0 / 3.0,
        "lift": (2.0 / 3.0) / 0.6,
        "leverage": 0.4 - (0.6 * 0.6),
        "conviction": (1.0 - 0.6) / (1.0 - 2.0 / 3.0),
    },
    (frozenset(["C"]), frozenset(["B"])): {
        "antecedent_support": 0.6,
        "consequent_support": 0.6,
        "support": 0.4,
        "confidence": 2.0 / 3.0,
        "lift": (2.0 / 3.0) / 0.6,
        "leverage": 0.4 - (0.6 * 0.6),
        "conviction": (1.0 - 0.6) / (1.0 - 2.0 / 3.0),
    },
    (frozenset(["B"]), frozenset(["E"])): {
        "antecedent_support": 0.6,
        "consequent_support": 0.4,
        "support": 0.4,
        "confidence": 2.0 / 3.0,
        "lift": (2.0 / 3.0) / 0.4,
        "leverage": 0.4 - (0.6 * 0.4),
        "conviction": (1.0 - 0.4) / (1.0 - 2.0 / 3.0),
    },
    (frozenset(["E"]), frozenset(["B"])): {
        "antecedent_support": 0.4,
        "consequent_support": 0.6,
        "support": 0.4,
        "confidence": 1.0,
        "lift": 1.0 / 0.6,
        "leverage": 0.4 - (0.4 * 0.6),
        "conviction": float("inf"),
    },
    (frozenset(["A", "B"]), frozenset(["C"])): {
        "antecedent_support": 0.4,
        "consequent_support": 0.6,
        "support": 0.4,
        "confidence": 1.0,
        "lift": 1.0 / 0.6,
        "leverage": 0.4 - (0.4 * 0.6),
        "conviction": float("inf"),
    },
    (frozenset(["B", "C"]), frozenset(["A"])): {
        "antecedent_support": 0.4,
        "consequent_support": 0.8,
        "support": 0.4,
        "confidence": 1.0,
        "lift": 1.0 / 0.8,
        "leverage": 0.4 - (0.4 * 0.8),
        "conviction": float("inf"),
    },
    (frozenset(["A", "C"]), frozenset(["B"])): {
        "antecedent_support": 0.6,
        "consequent_support": 0.6,
        "support": 0.4,
        "confidence": 2.0 / 3.0,
        "lift": (2.0 / 3.0) / 0.6,
        "leverage": 0.4 - (0.6 * 0.6),
        "conviction": (1.0 - 0.6) / (1.0 - 2.0 / 3.0),
    },
    (frozenset(["C"]), frozenset(["A", "B"])): {
        "antecedent_support": 0.6,
        "consequent_support": 0.4,
        "support": 0.4,
        "confidence": 2.0 / 3.0,
        "lift": (2.0 / 3.0) / 0.4,
        "leverage": 0.4 - (0.6 * 0.4),
        "conviction": (1.0 - 0.4) / (1.0 - 2.0 / 3.0),
    },
    (frozenset(["B"]), frozenset(["A", "C"])): {
        "antecedent_support": 0.6,
        "consequent_support": 0.6,
        "support": 0.4,
        "confidence": 2.0 / 3.0,
        "lift": (2.0 / 3.0) / 0.6,
        "leverage": 0.4 - (0.6 * 0.6),
        "conviction": (1.0 - 0.6) / (1.0 - 2.0 / 3.0),
    },
}
