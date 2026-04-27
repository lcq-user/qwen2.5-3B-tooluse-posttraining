#!/usr/bin/env python3
import argparse
import os
import shutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        default="third_party/When2Call/data",
        help="Path to the official When2Call data directory",
    )
    parser.add_argument(
        "--target-dir",
        default="data/raw/when2call",
        help="Project-local directory where the dataset should be copied",
    )
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)
    for split in ["train", "test"]:
        src = os.path.join(args.source_dir, split)
        dst = os.path.join(args.target_dir, split)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"Copied {src} -> {dst}")


if __name__ == "__main__":
    main()
