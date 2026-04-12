from aoede.runpod.dual_pod_launcher import (
    PodLaunchConfig,
    SharedWorkspaceConfig,
    _bootstrap_command,
    _network_volume_payload,
    _pod_payload,
)


def _shared_config() -> SharedWorkspaceConfig:
    return SharedWorkspaceConfig(
        workspace_mount="/workspace",
        root_repo_url="https://github.com/example/new_research.git",
        root_repo_branch="main",
        root_repo_dir_name="new_research",
        omnivoice_repo_url="https://github.com/example/OmniVoice.git",
        omnivoice_repo_branch="aoede-branch",
        cloneval_repo_url="https://github.com/amu-cai/cloneval.git",
        cloneval_repo_branch="main",
        bootstrap_script="scripts/runpod/bootstrap_workspace.sh",
        image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        data_center_id="US-KS-2",
        network_volume_name="aoede-omnivoice-shared",
        network_volume_size_gb=2048,
        container_disk_gb=80,
        vcpu_count=16,
        memory_in_gb=125,
        support_public_ip=True,
        hf_token="hf_secret",
    )


def test_bootstrap_command_clones_root_repo_and_runs_root_script():
    shared = _shared_config()
    command = _bootstrap_command(shared)
    assert "git clone --branch main https://github.com/example/new_research.git /workspace/new_research" in command
    assert "bash scripts/runpod/bootstrap_workspace.sh" in command


def test_pod_payload_includes_shared_workspace_env():
    shared = _shared_config()
    payload = _pod_payload(
        shared,
        PodLaunchConfig(name="aoede-train", gpu_type_id="NVIDIA A100 80GB PCIe", expose_jupyter=True),
        network_volume_id="vol_123",
    )
    assert payload["networkVolumeId"] == "vol_123"
    assert payload["env"]["ROOT_REPO_URL"] == "https://github.com/example/new_research.git"
    assert payload["env"]["OMNIVOICE_REPO_URL"] == "https://github.com/example/OmniVoice.git"
    assert payload["env"]["HF_TOKEN"] == "hf_secret"
    assert payload["ports"] == ["22/tcp", "8888/http"]


def test_network_volume_payload_matches_shared_config():
    shared = _shared_config()
    payload = _network_volume_payload(shared)
    assert payload == {
        "name": "aoede-omnivoice-shared",
        "size": 2048,
        "dataCenterId": "US-KS-2",
    }
