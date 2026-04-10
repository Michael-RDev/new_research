#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tokenize staged raw manifests and build a multilingual data config."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest_root",
        type=str,
        required=True,
        help="Directory containing raw JSONL manifests and *.summary.json sidecars.",
    )
    parser.add_argument(
        "--token_root",
        type=str,
        required=True,
        help="Directory where tokenized WebDataset shards should be written.",
    )
    parser.add_argument(
        "--data_config_out",
        type=str,
        required=True,
        help="Path to write the generated multilingual data config JSON.",
    )
    parser.add_argument(
        "--python_bin",
        type=str,
        default=sys.executable,
        help="Python executable used to call omnivoice.scripts.extract_audio_tokens.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="eustlb/higgs-audio-v2-tokenizer",
        help="Higgs audio tokenizer repo or local path.",
    )
    parser.add_argument(
        "--nj_per_gpu",
        type=int,
        default=3,
        help="Worker processes per GPU for token extraction.",
    )
    parser.add_argument(
        "--loader_workers",
        type=int,
        default=24,
        help="DataLoader workers used by the tokenization script.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="smoke",
        choices=["smoke", "core"],
        help="Controls repeat weighting in the generated data config.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print tokenization commands without executing them.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip manifests whose token output directory already has data.lst.",
    )
    return parser


def _discover_manifests(manifest_root: Path):
    manifests = []
    for manifest_path in sorted(manifest_root.glob("*.jsonl")):
        if manifest_path.name.endswith(".example.jsonl"):
            continue
        summary_path = manifest_path.with_suffix(".summary.json")
        if not summary_path.exists():
            continue
        manifests.append((manifest_path, summary_path))
    return manifests


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_command(
    python_bin: str,
    manifest_path: Path,
    token_dir: Path,
    tokenizer_path: str,
    nj_per_gpu: int,
    loader_workers: int,
):
    return [
        python_bin,
        "-m",
        "omnivoice.scripts.extract_audio_tokens",
        "--input_jsonl",
        str(manifest_path),
        "--tar_output_pattern",
        str(token_dir / "audios" / "shard-%06d.tar"),
        "--jsonl_output_pattern",
        str(token_dir / "txts" / "shard-%06d.jsonl"),
        "--tokenizer_path",
        tokenizer_path,
        "--nj_per_gpu",
        str(nj_per_gpu),
        "--loader_workers",
        str(loader_workers),
        "--shuffle",
        "True",
    ]


def _seconds_from_data_lst(path: Path):
    total = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 4:
            continue
        total += float(parts[3])
    return total


def _repeat_from_hours(hours: float, base_repeat: int, profile: str):
    if profile == "smoke":
        return 1
    if hours <= 10.0:
        return max(base_repeat, 8)
    if hours <= 25.0:
        return max(base_repeat, 4)
    if hours <= 50.0:
        return max(base_repeat, 2)
    return max(base_repeat, 1)


def _build_data_config(token_root: Path, data_config_out: Path, profile: str):
    grouped: dict[str, dict[tuple[str, str, int], list[str]]] = {
        "train": defaultdict(list),
        "dev": defaultdict(list),
    }

    for token_dir in sorted(token_root.iterdir()):
        if not token_dir.is_dir():
            continue
        manifest_path = token_dir / "data.lst"
        summary_path = token_dir / "manifest.summary.json"
        if not manifest_path.exists() or not summary_path.exists():
            continue

        summary = _read_json(summary_path)
        role = summary["split_role"]
        split_key = "dev" if role == "dev" else "train"
        hours = _seconds_from_data_lst(manifest_path) / 3600.0
        base_repeat = int(summary.get("repeat_hint", 1))
        repeat = (
            1
            if split_key == "dev"
            else _repeat_from_hours(
                hours=hours,
                base_repeat=base_repeat,
                profile=profile,
            )
        )
        dataset_name = str(summary["dataset_name"])
        language_id = str(summary["language_id"])
        grouped[split_key][(dataset_name, language_id, repeat)].append(str(manifest_path))

    payload = {"train": [], "dev": []}
    for split_key in ("train", "dev"):
        for (dataset_name, language_id, repeat), manifest_paths in sorted(grouped[split_key].items()):
            payload[split_key].append(
                {
                    "dataset_name": dataset_name,
                    "language_id": language_id,
                    "manifest_path": manifest_paths,
                    "repeat": repeat,
                }
            )

    data_config_out.parent.mkdir(parents=True, exist_ok=True)
    data_config_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    args = get_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    manifest_root = Path(args.manifest_root)
    token_root = Path(args.token_root)
    token_root.mkdir(parents=True, exist_ok=True)

    manifests = _discover_manifests(manifest_root)
    if not manifests:
        raise FileNotFoundError(f"No manifests with summary sidecars found in {manifest_root}")

    for manifest_path, summary_path in manifests:
        token_dir = token_root / manifest_path.stem
        data_lst_path = token_dir / "data.lst"
        if args.skip_existing and data_lst_path.exists():
            logging.info("Skipping existing token output for %s", manifest_path.name)
            continue

        cmd = _extract_command(
            python_bin=args.python_bin,
            manifest_path=manifest_path,
            token_dir=token_dir,
            tokenizer_path=args.tokenizer_path,
            nj_per_gpu=args.nj_per_gpu,
            loader_workers=args.loader_workers,
        )
        logging.info("Tokenizing %s -> %s", manifest_path.name, token_dir)
        logging.info("  %s", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
            shutil.copy2(summary_path, token_dir / "manifest.summary.json")

    if not args.dry_run:
        _build_data_config(
            token_root=token_root,
            data_config_out=Path(args.data_config_out),
            profile=args.profile,
        )
        logging.info("Wrote data config to %s", args.data_config_out)


if __name__ == "__main__":
    main()
