#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Provision and manage Runpod pods for MnemosVoice experiments."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from omnivoice.utils.env import load_env_file


REST_BASE_URL = "https://rest.runpod.io/v1"


@dataclass(frozen=True)
class RunpodLaunchConfig:
    name: str
    gpu_type_id: str
    data_center_id: str
    network_volume_name: str
    network_volume_size_gb: int
    image_name: str
    repo_url: str
    repo_branch: str
    cloneval_repo_url: str
    cloneval_repo_branch: str
    workspace_mount: str = "/workspace"
    repo_dir_name: str = "OmniVoice"
    container_disk_gb: int = 50
    vcpu_count: int = 16
    memory_in_gb: int = 125
    support_public_ip: bool = True
    expose_jupyter: bool = False
    hf_token: Optional[str] = None


class RunpodClient:
    def __init__(self, api_key: str, base_url: str = REST_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def _request(self, method: str, path: str, payload: Optional[dict[str, Any]] = None, params: Optional[dict[str, Any]] = None):
        import requests

        url = f"{self.base_url}{path}"
        response = requests.request(
            method=method,
            url=url,
            headers=self._headers(),
            json=payload,
            params=params,
            timeout=60,
        )
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()

    def list_pods(self, name: Optional[str] = None):
        params = {"name": name} if name else None
        data = self._request("GET", "/pods", params=params)
        return data.get("pods", data if isinstance(data, list) else [])

    def get_pod(self, pod_id: str):
        return self._request("GET", f"/pods/{pod_id}")

    def create_pod(self, payload: dict[str, Any]):
        return self._request("POST", "/pods", payload=payload)

    def start_pod(self, pod_id: str):
        return self._request("POST", f"/pods/{pod_id}/start")

    def delete_pod(self, pod_id: str):
        return self._request("DELETE", f"/pods/{pod_id}")

    def list_network_volumes(self):
        data = self._request("GET", "/networkvolumes")
        return data.get("networkVolumes", data if isinstance(data, list) else [])

    def create_network_volume(self, payload: dict[str, Any]):
        return self._request("POST", "/networkvolumes", payload=payload)


def _bootstrap_command(config: RunpodLaunchConfig):
    repo_path = f"{config.workspace_mount}/{config.repo_dir_name}"
    cloneval_path = f"{config.workspace_mount}/cloneval"
    return (
        "set -euo pipefail;"
        f" mkdir -p {config.workspace_mount};"
        f" if [ ! -d {repo_path}/.git ]; then git clone --branch {config.repo_branch} {config.repo_url} {repo_path}; fi;"
        f" if [ ! -d {cloneval_path}/.git ]; then git clone --branch {config.cloneval_repo_branch} {config.cloneval_repo_url} {cloneval_path}; fi;"
        f" cd {repo_path};"
        " bash examples/runpod/bootstrap_omnivoice.sh"
    )


def _pod_payload(config: RunpodLaunchConfig, network_volume_id: str):
    ports = ["22/tcp"]
    if config.expose_jupyter:
        ports.append("8888/http")

    env = {
        "REPO_URL": config.repo_url,
        "REPO_BRANCH": config.repo_branch,
        "CLONEVAL_REPO_URL": config.cloneval_repo_url,
        "CLONEVAL_REPO_BRANCH": config.cloneval_repo_branch,
        "WORKSPACE": config.workspace_mount,
    }
    if config.hf_token:
        env["HF_TOKEN"] = config.hf_token

    return {
        "cloudType": "SECURE",
        "computeType": "GPU",
        "containerDiskInGb": config.container_disk_gb,
        "dataCenterIds": [config.data_center_id],
        "dockerStartCmd": ["bash", "-lc", _bootstrap_command(config)],
        "env": env,
        "gpuCount": 1,
        "gpuTypeIds": [config.gpu_type_id],
        "imageName": config.image_name,
        "memoryInGb": config.memory_in_gb,
        "name": config.name,
        "networkVolumeId": network_volume_id,
        "ports": ports,
        "supportPublicIp": config.support_public_ip,
        "vcpuCount": config.vcpu_count,
        "volumeMountPath": config.workspace_mount,
    }


def _network_volume_payload(config: RunpodLaunchConfig):
    return {
        "name": config.network_volume_name,
        "size": config.network_volume_size_gb,
        "dataCenterId": config.data_center_id,
    }


def _find_volume_by_name(volumes: list[dict[str, Any]], name: str, data_center_id: str):
    for volume in volumes:
        if volume.get("name") == name and volume.get("dataCenterId") == data_center_id:
            return volume
    return None


def _build_launch_config(args, api_env: dict[str, str]):
    hf_token = api_env.get(args.hf_token_env) or os.environ.get(args.hf_token_env)
    volume_name = args.network_volume_name or f"{args.name}-workspace"
    return RunpodLaunchConfig(
        name=args.name,
        gpu_type_id=args.gpu_type_id,
        data_center_id=args.data_center_id,
        network_volume_name=volume_name,
        network_volume_size_gb=args.network_volume_size_gb,
        image_name=args.image_name,
        repo_url=args.repo_url,
        repo_branch=args.repo_branch,
        cloneval_repo_url=args.cloneval_repo_url,
        cloneval_repo_branch=args.cloneval_repo_branch,
        workspace_mount=args.workspace_mount,
        repo_dir_name=args.repo_dir_name,
        container_disk_gb=args.container_disk_gb,
        vcpu_count=args.vcpu_count,
        memory_in_gb=args.memory_in_gb,
        support_public_ip=not args.disable_public_ip,
        expose_jupyter=args.expose_jupyter,
        hf_token=hf_token,
    )


def _print_json(payload):
    print(json.dumps(payload, indent=2, sort_keys=True))


def _common_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--env_file", type=str, default=".env")
    parser.add_argument("--runpod_key_env", type=str, default="RUNPOD_API_KEY")
    parser.add_argument("--hf_token_env", type=str, default="HF_TOKEN")


def _add_launch_args(parser: argparse.ArgumentParser):
    parser.add_argument("--name", type=str, default="mnemosvoice-stage1")
    parser.add_argument("--gpu_type_id", type=str, default="NVIDIA A100 80GB PCIe")
    parser.add_argument("--data_center_id", type=str, default="US-KS-2")
    parser.add_argument("--network_volume_name", type=str, default=None)
    parser.add_argument("--network_volume_size_gb", type=int, default=2048)
    parser.add_argument(
        "--image_name",
        type=str,
        default="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    )
    parser.add_argument("--repo_url", type=str, required=True)
    parser.add_argument("--repo_branch", type=str, default="master")
    parser.add_argument(
        "--cloneval_repo_url",
        type=str,
        default="https://github.com/amu-cai/cloneval.git",
    )
    parser.add_argument("--cloneval_repo_branch", type=str, default="main")
    parser.add_argument("--workspace_mount", type=str, default="/workspace")
    parser.add_argument("--repo_dir_name", type=str, default="OmniVoice")
    parser.add_argument("--container_disk_gb", type=int, default=50)
    parser.add_argument("--vcpu_count", type=int, default=16)
    parser.add_argument("--memory_in_gb", type=int, default=125)
    parser.add_argument("--disable_public_ip", action="store_true")
    parser.add_argument("--expose_jupyter", action="store_true")
    parser.add_argument("--dry_run", action="store_true")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a Runpod training pod.")
    _common_parser(create_parser)
    _add_launch_args(create_parser)

    status_parser = subparsers.add_parser("status", help="Inspect a pod or list pods by name.")
    _common_parser(status_parser)
    status_parser.add_argument("--pod_id", type=str, default=None)
    status_parser.add_argument("--name", type=str, default=None)

    start_parser = subparsers.add_parser("start", help="Start an existing pod.")
    _common_parser(start_parser)
    start_parser.add_argument("--pod_id", type=str, required=True)

    delete_parser = subparsers.add_parser("terminate", help="Terminate a pod.")
    _common_parser(delete_parser)
    delete_parser.add_argument("--pod_id", type=str, required=True)

    args = parser.parse_args()
    env_values = load_env_file(args.env_file) if getattr(args, "env_file", None) else {}

    if args.command == "create":
        config = _build_launch_config(args, env_values)
        volume_payload = _network_volume_payload(config)
        pod_payload = _pod_payload(config, network_volume_id="<create-me>")

        if args.dry_run:
            _print_json(
                {
                    "launch_config": asdict(config),
                    "network_volume_payload": volume_payload,
                    "pod_payload": pod_payload,
                }
            )
            return

    api_key = env_values.get(args.runpod_key_env) or os.environ.get(args.runpod_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing {args.runpod_key_env}. Provide it in {args.env_file} or the environment."
        )

    client = RunpodClient(api_key=api_key)

    if args.command == "create":
        existing_volume = _find_volume_by_name(
            client.list_network_volumes(),
            name=config.network_volume_name,
            data_center_id=config.data_center_id,
        )
        if existing_volume:
            network_volume = existing_volume
        else:
            network_volume = client.create_network_volume(volume_payload)

        pod_payload = _pod_payload(config, network_volume_id=network_volume["id"])

        created = client.create_pod(pod_payload)
        _print_json(
            {
                "network_volume": network_volume,
                "pod": created,
            }
        )
        return

    if args.command == "status":
        if args.pod_id:
            _print_json(client.get_pod(args.pod_id))
        else:
            _print_json({"pods": client.list_pods(name=args.name)})
        return

    if args.command == "start":
        _print_json(client.start_pod(args.pod_id))
        return

    if args.command == "terminate":
        _print_json(client.delete_pod(args.pod_id))
        return


if __name__ == "__main__":
    main()
