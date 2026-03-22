#!/usr/bin/env python3
"""
Download NSL-KDD dataset from GitHub mirror.
Run: python scripts/download_data.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import NSLKDDLoader


def main():
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    print(f"Downloading NSL-KDD to {data_dir}")
    NSLKDDLoader.download_from_github(str(data_dir))
    print("Done. Verify KDDTrain+.txt and KDDTest+.txt exist.")


if __name__ == "__main__":
    main()
