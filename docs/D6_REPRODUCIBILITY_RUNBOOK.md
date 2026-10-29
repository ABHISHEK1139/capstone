# Deliverable D6: Reproducibility Runbook & User Operating Guide

**Capstone Project ID**: KLCAP-2026-00340  
**Title**: Scalable Association-Rule Mining Estimator with Apriori Baseline  

---

## 1. Environment & Prerequisites

- **Python Version**: 3.10+ (Developed and certified on **Python 3.12.10**)
- **Platform**: Cross-platform (Windows, Linux, macOS)
- **Core Dependencies**:
  - `numpy >= 1.24.0`
  - `scipy >= 1.10.0`
  - `pandas >= 2.0.0`
  - `scikit-learn >= 1.3.0`
  - `joblib >= 1.2.0`
  - `pytest >= 7.0.0`
  - `psutil >= 5.9.0`

---

## 2. Installation & Quickstart

### Step 1: Clone and Setup Workspace
```bash
cd capstone
python -m venv .venv
# Activate virtual environment:
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies:
pip install -r requirements.txt
```

### Step 2: Run Full Automated Test Suite (25 Tests)
```bash
pytest -v
```

### Step 3: Run Independent Acceptance Harness (AC-1 to AC-4, NT-1 to NT-6)
```bash
python tests/run_acceptance.py
```

### Step 4: Run Scalability & KPI Benchmark Suite
```bash
python benchmarks/run_benchmarks.py
```

### Step 5: Run Interactive CLI Demonstration
```bash
python -m skarm.cli --demo
```

### Step 6: Download Benchmark Datasets (Groceries, Chess, Retail)
```bash
python scripts/download_datasets.py
```

### Step 7: Run End-to-End Real-World Dataset Benchmarks
```bash
python benchmarks/run_real_benchmarks.py
```

### Step 8: Mine Real Market Basket Association Rules via CLI
```bash
# Mine Groceries dataset with lift ranking:
python -m skarm.cli --dataset groceries --min-support 0.02 --min-confidence 0.25 --top-n 10

# Mine Retail dataset (88,162 transactions):
python -m skarm.cli --dataset retail --min-support 0.02 --min-confidence 0.30 --top-n 10 --algorithm eclat
```

---

## 3. Python API Usage Examples

### Example 1: Basic Market Basket Analysis with Apriori Baseline
```python
from skarm.estimator import AssociationRulesEstimator

transactions = [
    ["milk", "bread", "butter"],
    ["bread", "butter"],
    ["milk", "bread"],
    ["beer", "diapers"],
    ["milk", "beer", "diapers", "bread"],
    ["bread", "butter", "milk"],
    ["diapers", "beer"],
    ["milk", "bread", "butter", "eggs"],
]

# Initialize estimator
estimator = AssociationRulesEstimator(
    min_support=0.3,
    metric="confidence",
    min_threshold=0.6,
    algorithm="apriori",
)

# Fit on transactions
estimator.fit(transactions)

# Inspect discovered frequent itemsets and rules
print(estimator.get_frequent_itemsets())
print(estimator.get_rules()[["antecedent_names", "consequent_names", "support", "confidence", "lift"]])

# Predict recommendations for a new customer basket
new_basket = [["milk", "bread"]]
recommendations = estimator.predict(new_basket)
print("Recommended add-ons:", recommendations)
# Output: [['butter']]
```

### Example 2: Scikit-Learn Pipeline and Joblib Serialization
```python
import joblib
from skarm.estimator import AssociationRulesEstimator

estimator = AssociationRulesEstimator(
    min_support=0.2,
    metric="lift",
    min_threshold=1.2,
    algorithm="fpgrowth",
)
estimator.fit(transactions)

# Serialize model to disk
joblib.dump(estimator, "association_rules_model.joblib")

# Deserialize model
loaded_model = joblib.load("association_rules_model.joblib")
print("Model loaded successfully. Discovered rules count:", len(loaded_model.rules_))
```

### Example 3: Handling Sparse Matrices with Eclat
```python
from scipy import sparse
from skarm.preprocessing import TransactionEncoder
from skarm.estimator import AssociationRulesEstimator

# Encode transactions into CSR sparse matrix
encoder = TransactionEncoder(sparse_output=True)
X_sparse = encoder.fit_transform(transactions)

# Fit estimator directly on sparse matrix using Eclat vertical miner
sparse_estimator = AssociationRulesEstimator(
    min_support=0.25,
    algorithm="eclat",
)
sparse_estimator.fit(X_sparse)
print("Sparse rules discovered:", len(sparse_estimator.rules_))
```
