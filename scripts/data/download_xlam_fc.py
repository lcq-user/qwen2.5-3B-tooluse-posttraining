#!/usr/bin/env python3
import argparse
import os
from modelscope.msdatasets import MsDataset

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default='Salesforce/xlam-function-calling-60k',
    )
    parser.add_argument("--output-dir", default="data/raw/xlam_fc_60k")
    parser.add_argument("--filename", default="xlam_fc_60k.jsonl")
    args = parser.parse_args()

    ds = MsDataset.load(args.repo_id, subset_name='dataset', split='train')
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.filename)
    
    ds.to_json(output_path)
    print(f"save to {output_path}")

if __name__ == "__main__":
    main()
