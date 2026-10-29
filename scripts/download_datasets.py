"""Download benchmark datasets for Association Rule Mining.
Datasets:
1. Groceries (9,835 transactions, supermarket basket data)
2. Chess (3,196 transactions, dense benchmark)
3. Retail (88,162 transactions, sparse market basket benchmark)
"""
import os
import urllib.request
import time

DATASETS = {
    "groceries.csv": "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv",
    "chess.dat": "https://raw.githubusercontent.com/SarraLKSC/Minning-patterns-in-data/master/chess.dat",
    "retail.dat": "https://raw.githubusercontent.com/ihewro/Apriori/master/retail.dat",
    "market_basket.csv": "https://raw.githubusercontent.com/sharmaroshan/Market-Basket-Analysis/master/Market_Basket_Optimisation.csv",
    "mushroom.dat": "https://raw.githubusercontent.com/hyperair/fptree/master/datasets/mushroom.dat",
}

def download_file(name: str, url: str, target_dir: str = "data"):
    target_path = os.path.join(target_dir, name)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        print(f"[SKIP] {name} already exists ({os.path.getsize(target_path):,} bytes).")
        return target_path

    print(f"[DOWNLOADING] {name} from {url}...")
    t0 = time.time()
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ARM-Capstone-Evaluator/1.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp, open(target_path, "wb") as f:
        while chunk := resp.read(65536):
            f.write(chunk)
    elapsed = time.time() - t0
    size = os.path.getsize(target_path)
    print(f"[OK] Downloaded {name}: {size:,} bytes in {elapsed:.2f}s.")
    return target_path

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    for name, url in DATASETS.items():
        try:
            download_file(name, url)
        except Exception as e:
            print(f"[ERROR] Failed downloading {name}: {e}")
