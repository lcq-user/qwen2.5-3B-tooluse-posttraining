#!/usr/bin/env python3
import argparse
import os
import shutil
import fnmatch

from huggingface_hub import hf_hub_download, list_repo_files
from modelscope.hub.api import HubApi
from modelscope.hub.file_download import dataset_file_download


def download_with_huggingface(
    repo_id: str,
    target_dir: str,
    include: list[str],
    endpoint: str | None,
) -> None:
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint

    files = list_repo_files(repo_id, repo_type="dataset")
    wanted = [
        name for name in files
        if any(fnmatch.fnmatch(name, pattern) for pattern in include)
    ]
    if not wanted:
        raise RuntimeError(
            f"No dataset files matched {include} in {repo_id}"
        )

    for filename in wanted:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
        )
        dst = os.path.join(target_dir, os.path.basename(filename))
        shutil.copy2(local_path, dst)
        print(f"Downloaded {filename} -> {dst}")


def _extract_modelscope_path(entry: object) -> str | None:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("Path") or entry.get("path")
    return getattr(entry, "Path", None) or getattr(entry, "path", None)


def download_with_modelscope(
    repo_id: str,
    target_dir: str,
    include: list[str],
    revision: str,
) -> None:
    api = HubApi()
    files = api.get_dataset_files(repo_id=repo_id, revision=revision)
    wanted = []
    for entry in files:
        path = _extract_modelscope_path(entry)
        if path and any(fnmatch.fnmatch(path, pattern) for pattern in include):
            wanted.append(path)
    if not wanted:
        raise RuntimeError(
            f"No dataset files matched {include} in {repo_id} via ModelScope"
        )

    for filename in wanted:
        local_path = dataset_file_download(
            dataset_id=repo_id,
            file_path=filename,
            revision=revision,
        )
        dst = os.path.join(target_dir, os.path.basename(filename))
        shutil.copy2(local_path, dst)
        print(f"Downloaded {filename} -> {dst}")


def download_ai_modelscope_bfcl_v3(
    repo_id: str,
    target_dir: str,
    revision: str,
) -> None:
    files = [
        "README.md",
        "dataset_infos.json",
        "data/train-00000-of-00001.parquet",
    ]
    for filename in files:
        local_path = dataset_file_download(
            dataset_id=repo_id,
            file_path=filename,
            revision=revision,
        )
        dst_name = os.path.basename(filename)
        dst = os.path.join(target_dir, dst_name)
        shutil.copy2(local_path, dst)
        print(f"Downloaded {filename} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default="AI-ModelScope/bfcl_v3",
    )
    parser.add_argument(
        "--target-dir",
        default="data/raw/bfcl",
    )
    parser.add_argument(
        "--source",
        choices=["modelscope", "huggingface", "modelscope_parquet"],
        default="modelscope",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Optional Hugging Face endpoint, e.g. https://hf-mirror.com",
    )
    parser.add_argument(
        "--revision",
        default="master",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=["BFCL_v3*.json", "BFCL_v3*.jsonl"],
        help="Glob patterns for files to download",
    )
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)
    if args.source == "modelscope_parquet":
        download_ai_modelscope_bfcl_v3(
            repo_id=args.repo_id,
            target_dir=args.target_dir,
            revision=args.revision,
        )
    elif args.source == "modelscope":
        download_with_modelscope(
            repo_id=args.repo_id,
            target_dir=args.target_dir,
            include=args.include,
            revision=args.revision,
        )
    else:
        download_with_huggingface(
            repo_id=args.repo_id,
            target_dir=args.target_dir,
            include=args.include,
            endpoint=args.endpoint,
        )


if __name__ == "__main__":
    main()
