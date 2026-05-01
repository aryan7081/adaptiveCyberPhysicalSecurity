#!/usr/bin/env python3
"""Download data helper (NSL-KDD auto; CIC-IDS manual)."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.data.loader import NSLKDDLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dataset_name = str(cfg.get("dataset", {}).get("name", "nsl_kdd")).lower()
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    if dataset_name in {"nsl_kdd", "nsl-kdd", "nsl"}:
        print(f"Downloading NSL-KDD to {data_dir}")
        NSLKDDLoader.download_from_github(str(data_dir))
        print("Done. Verify KDDTrain+.txt and KDDTest+.txt exist.")
        return
    if dataset_name in {"cic_ids", "cic-ids", "cicids2017", "cic_ids_2017"}:
        print("CIC-IDS download is manual due to dataset size and mirror variability.")
        print(f"Place CSV files in: {data_dir}")
        print(f"Expected train file: {cfg['dataset'].get('train_file')}")
        print(f"Expected test file: {cfg['dataset'].get('test_file')}")
        return
    raise ValueError(f"Unsupported dataset.name: {dataset_name}")


if __name__ == "__main__":
    main()
