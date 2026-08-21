"""Download the public UCI dataset required by this demonstration."""
from pathlib import Path
import sys
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_PATH

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"


def main():
    DATA_PATH.parent.mkdir(exist_ok=True)
    urlretrieve(URL, DATA_PATH)
    print(f"Downloaded dataset to {DATA_PATH}")


if __name__ == "__main__":
    main()
