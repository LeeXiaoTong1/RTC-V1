#!/usr/bin/env python3
"""Prepare validated training pairs from the official correspondence CSV."""

import argparse
import json

from utils.rtc_pairs import prepare_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs_csv", required=True, help="Official train_offline_online_pairs.csv")
    parser.add_argument("--train_protocol", required=True)
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--output", required=True, help="Output JSONL manifest")
    parser.add_argument("--offline_column", help="Exact CSV header if auto-detection is ambiguous")
    parser.add_argument("--online_column", help="Exact CSV header if auto-detection is ambiguous")
    args = parser.parse_args()
    try:
        _, metadata = prepare_pairs(**vars(args))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Pair preparation failed: {exc}\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Saved validated pairs: {args.output}")
    print("Unpaired training audio stays in the main V1 classification loader.")


if __name__ == "__main__":
    main()
