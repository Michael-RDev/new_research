from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional
from urllib import error, parse, request


REST_BASE_URL = "https://rest.runpod.io/v1"


def parse_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    env_path = Path(path)
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            values[key] = value
    return values


@dataclass(frozen=True)
class SharedWorkspaceConfig:
    workspace_mount: str
    root_repo_url: str
    root_repo_branch: str
    root_repo_dir_name: str
    omnivoice_repo_url: str
    omnivoice_repo_branch: str
    cloneval_repo_url: str
    cloneval_repo_branch: str
    bootstrap_script: str
    image_name: str
    data_center_id: str
    network_volume_name: str
    network_volume_size_gb: int
    container_disk_gb: int
    pod_volume_gb: int
    vcpu_count: int
    memory_in_gb: int
    support_public_ip: bool
    hf_token: Optional[str] = None


@dataclass(frozen=True)
class PodLaunchConfig:
    name: str
    gpu_type_id: str
    expose_jupyter: bool = False


class RunpodClient:
    def __init__(self, api_key: str, base_url: str = REST_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
    ):
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{parse.urlencode(params)}"
        body = None
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(url=url, data=body, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=60) as response:
                raw = response.read()
        except error.HTTPError as exc:  # pragma: no cover - exercised only via live API
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"RunPod API {method} {url} failed: {exc.code} {detail}") from exc
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def list_pods(self, name: Optional[str] = None):
        params = {"name": name} if name else None
        data = self._request("GET", "/pods", params=params)
        if isinstance(data, list):
            return data
        return data.get("pods", [])

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
        if isinstance(data, list):
            return data
        return data.get("networkVolumes", [])

    def create_network_volume(self, payload: dict[str, Any]):
        return self._request("POST", "/networkvolumes", payload=payload)


def _root_repo_path(config: SharedWorkspaceConfig) -> str:
    return f"{config.workspace_mount}/{config.root_repo_dir_name}"


def _bootstrap_command(config: SharedWorkspaceConfig) -> str:
    root_repo_path = _root_repo_path(config)
    return (
        "set -euo pipefail;"
        f" mkdir -p {config.workspace_mount};"
        f" if [ ! -d {root_repo_path}/.git ]; then git clone --branch {config.root_repo_branch} {config.root_repo_url} {root_repo_path}; fi;"
        f" cd {root_repo_path};"
        f" bash {config.bootstrap_script}"
    )


def _pod_payload(
    shared: SharedWorkspaceConfig,
    pod: PodLaunchConfig,
    network_volume_id: Optional[str],
) -> dict[str, Any]:
    ports = ["22/tcp"]
    if pod.expose_jupyter:
        ports.append("8888/http")

    env = {
        "WORKSPACE": shared.workspace_mount,
        "ROOT_REPO_URL": shared.root_repo_url,
        "ROOT_REPO_BRANCH": shared.root_repo_branch,
        "ROOT_REPO_DIR_NAME": shared.root_repo_dir_name,
        "OMNIVOICE_REPO_URL": shared.omnivoice_repo_url,
        "OMNIVOICE_REPO_BRANCH": shared.omnivoice_repo_branch,
        "CLONEVAL_REPO_URL": shared.cloneval_repo_url,
        "CLONEVAL_REPO_BRANCH": shared.cloneval_repo_branch,
    }
    if shared.hf_token:
        env["HF_TOKEN"] = shared.hf_token

    payload = {
        "cloudType": "SECURE",
        "computeType": "GPU",
        "containerDiskInGb": shared.container_disk_gb,
        "dataCenterIds": [shared.data_center_id],
        "dockerStartCmd": ["bash", "-lc", _bootstrap_command(shared)],
        "env": env,
        "gpuCount": 1,
        "gpuTypeIds": [pod.gpu_type_id],
        "imageName": shared.image_name,
        "minRAMPerGPU": shared.memory_in_gb,
        "minVCPUPerGPU": shared.vcpu_count,
        "name": pod.name,
        "ports": ports,
        "supportPublicIp": shared.support_public_ip,
        "volumeMountPath": shared.workspace_mount,
    }
    if network_volume_id:
        payload["networkVolumeId"] = network_volume_id
    else:
        payload["volumeInGb"] = shared.pod_volume_gb
    return payload


def _network_volume_payload(shared: SharedWorkspaceConfig) -> dict[str, Any]:
    return {
        "name": shared.network_volume_name,
        "size": shared.network_volume_size_gb,
        "dataCenterId": shared.data_center_id,
    }


def _find_volume_by_name(
    volumes: list[dict[str, Any]],
    name: str,
    data_center_id: str,
):
    for volume in volumes:
        if volume.get("name") == name and volume.get("dataCenterId") == data_center_id:
            return volume
    return None


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _common_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--env_file", type=str, default=".env")
    parser.add_argument("--runpod_key_env", type=str, default="RUNPOD_API_KEY")
    parser.add_argument("--hf_token_env", type=str, default="HF_TOKEN")


def _add_shared_args(parser: argparse.ArgumentParser):
    parser.add_argument("--workspace_mount", type=str, default="/workspace")
    parser.add_argument("--root_repo_url", type=str, required=True)
    parser.add_argument("--root_repo_branch", type=str, default="main")
    parser.add_argument("--root_repo_dir_name", type=str, default="new_research")
    parser.add_argument(
        "--omnivoice_repo_url",
        type=str,
        default="https://github.com/k2-fsa/OmniVoice.git",
    )
    parser.add_argument("--omnivoice_repo_branch", type=str, default="master")
    parser.add_argument(
        "--cloneval_repo_url",
        type=str,
        default="https://github.com/amu-cai/cloneval.git",
    )
    parser.add_argument("--cloneval_repo_branch", type=str, default="main")
    parser.add_argument(
        "--bootstrap_script",
        type=str,
        default="scripts/runpod/bootstrap_workspace.sh",
    )
    parser.add_argument(
        "--image_name",
        type=str,
        default="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    )
    parser.add_argument("--data_center_id", type=str, default="US-KS-2")
    parser.add_argument("--network_volume_name", type=str, default="aoede-omnivoice-shared")
    parser.add_argument("--network_volume_size_gb", type=int, default=2048)
    parser.add_argument("--container_disk_gb", type=int, default=80)
    parser.add_argument("--pod_volume_gb", type=int, default=200)
    parser.add_argument("--vcpu_count", type=int, default=16)
    parser.add_argument("--memory_in_gb", type=int, default=125)
    parser.add_argument("--disable_public_ip", action="store_true")
    parser.add_argument("--skip_network_volume", action="store_true")


def _add_pod_args(parser: argparse.ArgumentParser):
    parser.add_argument("--train_name", type=str, default="aoede-train")
    parser.add_argument("--train_gpu_type_id", type=str, default="NVIDIA A100 80GB PCIe")
    parser.add_argument("--train_expose_jupyter", action="store_true")
    parser.add_argument("--eval_name", type=str, default="omnivoice-eval")
    parser.add_argument("--eval_gpu_type_id", type=str, default="NVIDIA A100 80GB PCIe")
    parser.add_argument("--eval_expose_jupyter", action="store_true")
    parser.add_argument("--dry_run", action="store_true")


def _build_shared_config(args, env_values: dict[str, str]) -> SharedWorkspaceConfig:
    hf_token = env_values.get(args.hf_token_env) or os.environ.get(args.hf_token_env)
    return SharedWorkspaceConfig(
        workspace_mount=args.workspace_mount,
        root_repo_url=args.root_repo_url,
        root_repo_branch=args.root_repo_branch,
        root_repo_dir_name=args.root_repo_dir_name,
        omnivoice_repo_url=args.omnivoice_repo_url,
        omnivoice_repo_branch=args.omnivoice_repo_branch,
        cloneval_repo_url=args.cloneval_repo_url,
        cloneval_repo_branch=args.cloneval_repo_branch,
        bootstrap_script=args.bootstrap_script,
        image_name=args.image_name,
        data_center_id=args.data_center_id,
        network_volume_name=args.network_volume_name,
        network_volume_size_gb=args.network_volume_size_gb,
        container_disk_gb=args.container_disk_gb,
        pod_volume_gb=args.pod_volume_gb,
        vcpu_count=args.vcpu_count,
        memory_in_gb=args.memory_in_gb,
        support_public_ip=not args.disable_public_ip,
        hf_token=hf_token,
    )


def _build_pod_configs(args) -> tuple[PodLaunchConfig, PodLaunchConfig]:
    train = PodLaunchConfig(
        name=args.train_name,
        gpu_type_id=args.train_gpu_type_id,
        expose_jupyter=args.train_expose_jupyter,
    )
    eval_pod = PodLaunchConfig(
        name=args.eval_name,
        gpu_type_id=args.eval_gpu_type_id,
        expose_jupyter=args.eval_expose_jupyter,
    )
    return train, eval_pod


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision two RunPod pods for Aoede training and OmniVoice evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create the shared volume plus train/eval pods.")
    _common_parser(create_parser)
    _add_shared_args(create_parser)
    _add_pod_args(create_parser)

    status_parser = subparsers.add_parser("status", help="Inspect train/eval pods by id or by default names.")
    _common_parser(status_parser)
    status_parser.add_argument("--train_pod_id", type=str, default=None)
    status_parser.add_argument("--eval_pod_id", type=str, default=None)
    status_parser.add_argument("--train_name", type=str, default="aoede-train")
    status_parser.add_argument("--eval_name", type=str, default="omnivoice-eval")

    start_parser = subparsers.add_parser("start", help="Start existing train/eval pods.")
    _common_parser(start_parser)
    start_parser.add_argument("--train_pod_id", type=str, required=True)
    start_parser.add_argument("--eval_pod_id", type=str, required=True)

    terminate_parser = subparsers.add_parser("terminate", help="Terminate existing train/eval pods.")
    _common_parser(terminate_parser)
    terminate_parser.add_argument("--train_pod_id", type=str, required=True)
    terminate_parser.add_argument("--eval_pod_id", type=str, required=True)

    args = parser.parse_args()
    env_values = parse_env_file(args.env_file) if getattr(args, "env_file", None) else {}

    if args.command == "create":
        shared = _build_shared_config(args, env_values)
        train_pod, eval_pod = _build_pod_configs(args)
        volume_payload = _network_volume_payload(shared)
        planned_network_volume_id = None if args.skip_network_volume else "<create-me>"
        train_payload = _pod_payload(shared, train_pod, network_volume_id=planned_network_volume_id)
        eval_payload = _pod_payload(shared, eval_pod, network_volume_id=planned_network_volume_id)
        if args.dry_run:
            _print_json(
                {
                    "shared_workspace": asdict(shared),
                    "network_volume_payload": None if args.skip_network_volume else volume_payload,
                    "train_pod_payload": train_payload,
                    "eval_pod_payload": eval_payload,
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
        network_volume = None
        if not args.skip_network_volume:
            existing_volume = _find_volume_by_name(
                client.list_network_volumes(),
                name=shared.network_volume_name,
                data_center_id=shared.data_center_id,
            )
            network_volume = existing_volume or client.create_network_volume(volume_payload)
        train_payload = _pod_payload(
            shared,
            train_pod,
            network_volume_id=None if network_volume is None else network_volume["id"],
        )
        eval_payload = _pod_payload(
            shared,
            eval_pod,
            network_volume_id=None if network_volume is None else network_volume["id"],
        )
        created_train = client.create_pod(train_payload)
        created_eval = client.create_pod(eval_payload)
        _print_json(
            {
                "network_volume": network_volume,
                "train_pod": created_train,
                "eval_pod": created_eval,
            }
        )
        return

    if args.command == "status":
        train_status = client.get_pod(args.train_pod_id) if args.train_pod_id else {"pods": client.list_pods(name=args.train_name)}
        eval_status = client.get_pod(args.eval_pod_id) if args.eval_pod_id else {"pods": client.list_pods(name=args.eval_name)}
        _print_json({"train": train_status, "eval": eval_status})
        return

    if args.command == "start":
        _print_json(
            {
                "train": client.start_pod(args.train_pod_id),
                "eval": client.start_pod(args.eval_pod_id),
            }
        )
        return

    if args.command == "terminate":
        _print_json(
            {
                "train": client.delete_pod(args.train_pod_id),
                "eval": client.delete_pod(args.eval_pod_id),
            }
        )


if __name__ == "__main__":
    main()
