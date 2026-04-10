#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Build staged raw JSONL manifests for OmniVoice / MnemosVoice training."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from omnivoice.data.manifest_builder import (
    build_hf_manifests,
    stage_recipes,
    write_manifest_plan,
)
from omnivoice.utils.env import load_env_file


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        type=str,
        default="stage1",
        choices=["stage1", "stage2", "all"],
        help="Which staged corpus recipe to materialize.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="smoke",
        choices=["smoke", "core"],
        help="Smoke keeps tiny subsets; core uses the planned stage-1 caps.",
    )
    parser.add_argument(
        "--manifest_root",
        type=str,
        required=True,
        help="Directory for output JSONL manifests and summary sidecars.",
    )
    parser.add_argument(
        "--audio_root",
        type=str,
        required=True,
        help="Directory where streamed audio files are materialized.",
    )
    parser.add_argument(
        "--env_file",
        type=str,
        default=".env",
        help="Optional .env file that may contain HF_TOKEN.",
    )
    parser.add_argument(
        "--hf_token_env",
        type=str,
        default="HF_TOKEN",
        help="Environment variable name used to read the Hugging Face token.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip writing rows when the output JSONL already exists.",
    )
    parser.add_argument(
        "--write_recipe_plan",
        action="store_true",
        help="Write a JSON dump of the active recipe list next to the manifests.",
    )
    return parser


def main():
    args = get_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.env_file:
        load_env_file(args.env_file)

    manifest_root = Path(args.manifest_root)
    audio_root = Path(args.audio_root)
    manifest_root.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)

    recipes = stage_recipes(stage=args.stage, profile=args.profile)
    hf_token = os.environ.get(args.hf_token_env)

    if args.write_recipe_plan:
        write_manifest_plan(
            recipes,
            str(manifest_root / f"{args.stage}_{args.profile}_recipes.json"),
        )

    built = build_hf_manifests(
        recipes=recipes,
        manifest_root=str(manifest_root),
        audio_root=str(audio_root),
        hf_token=hf_token,
        skip_existing=args.skip_existing,
    )

    logging.info("Built %s manifest files under %s", len(built), manifest_root)
    for path in built:
        logging.info("  %s", path)


if __name__ == "__main__":
    main()

