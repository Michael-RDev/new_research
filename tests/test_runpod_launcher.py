from omnivoice.runpod.launcher import (
    RunpodLaunchConfig,
    _bootstrap_command,
    _network_volume_payload,
    _pod_payload,
)


def test_pod_payload_contains_bootstrap_and_env():
    config = RunpodLaunchConfig(
        name="mnemosvoice-stage1",
        gpu_type_id="NVIDIA A100 80GB PCIe",
        data_center_id="US-KS-2",
        network_volume_name="mnemosvoice-stage1-workspace",
        network_volume_size_gb=2048,
        image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        repo_url="https://github.com/example/OmniVoice.git",
        repo_branch="mnemosvoice",
        cloneval_repo_url="https://github.com/amu-cai/cloneval.git",
        cloneval_repo_branch="main",
        hf_token="hf_secret",
        expose_jupyter=True,
    )

    payload = _pod_payload(config, network_volume_id="vol_123")
    assert payload["networkVolumeId"] == "vol_123"
    assert payload["gpuTypeIds"] == ["NVIDIA A100 80GB PCIe"]
    assert payload["ports"] == ["22/tcp", "8888/http"]
    assert payload["env"]["HF_TOKEN"] == "hf_secret"
    assert "bootstrap_omnivoice.sh" in _bootstrap_command(config)


def test_network_volume_payload_uses_requested_size_and_datacenter():
    config = RunpodLaunchConfig(
        name="mnemosvoice-stage1",
        gpu_type_id="NVIDIA A100 80GB PCIe",
        data_center_id="US-KS-2",
        network_volume_name="mnemosvoice-stage1-workspace",
        network_volume_size_gb=2048,
        image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        repo_url="https://github.com/example/OmniVoice.git",
        repo_branch="mnemosvoice",
        cloneval_repo_url="https://github.com/amu-cai/cloneval.git",
        cloneval_repo_branch="main",
    )

    payload = _network_volume_payload(config)
    assert payload["name"] == "mnemosvoice-stage1-workspace"
    assert payload["size"] == 2048
    assert payload["dataCenterId"] == "US-KS-2"

