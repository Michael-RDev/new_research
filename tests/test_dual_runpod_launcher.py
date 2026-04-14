from aoede.runpod.dual_pod_launcher import (
    PodLaunchConfig,
    RunpodClient,
    SharedWorkspaceConfig,
    _bootstrap_command,
    _create_pod_with_fallbacks,
    _network_volume_payload,
    _pod_payload,
    _resource_fallbacks,
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
        cloud_type="SECURE",
        data_center_ids=("US-KS-2",),
        network_volume_name="aoede-omnivoice-shared",
        network_volume_size_gb=2048,
        container_disk_gb=80,
        pod_volume_gb=200,
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
    assert "RunPod bootstrap failed" in command
    assert "exec sleep infinity" in command


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
    assert payload["minRAMPerGPU"] == 125
    assert payload["minVCPUPerGPU"] == 16
    assert payload["dataCenterIds"] == ["US-KS-2"]
    assert payload["cloudType"] == "SECURE"


def test_pod_payload_uses_local_volume_when_network_volume_is_disabled():
    shared = _shared_config()
    payload = _pod_payload(
        shared,
        PodLaunchConfig(name="aoede-train", gpu_type_id="NVIDIA A100 80GB PCIe"),
        network_volume_id=None,
    )
    assert "networkVolumeId" not in payload
    assert payload["volumeInGb"] == 200


def test_network_volume_payload_matches_shared_config():
    shared = _shared_config()
    payload = _network_volume_payload(shared)
    assert payload == {
        "name": "aoede-omnivoice-shared",
        "size": 2048,
        "dataCenterId": "US-KS-2",
    }


def test_pod_payload_omits_data_centers_when_any_data_center_is_enabled():
    shared = SharedWorkspaceConfig(
        **{**_shared_config().__dict__, "cloud_type": "COMMUNITY", "data_center_ids": ()}
    )
    payload = _pod_payload(
        shared,
        PodLaunchConfig(name="omnivoice-eval", gpu_type_id="NVIDIA RTX A4500"),
        network_volume_id=None,
    )
    assert "dataCenterIds" not in payload
    assert payload["cloudType"] == "COMMUNITY"


def test_client_accepts_top_level_list_responses(monkeypatch):
    client = RunpodClient(api_key="rp_test")

    monkeypatch.setattr(
        client,
        "_request",
        lambda method, path, payload=None, params=None: [{"id": "pod_123"}] if path == "/pods" else [{"id": "vol_123"}],
    )

    assert client.list_pods() == [{"id": "pod_123"}]
    assert client.list_network_volumes() == [{"id": "vol_123"}]


def test_resource_fallbacks_include_smaller_shapes():
    fallbacks = _resource_fallbacks(_shared_config())
    assert fallbacks[0] == (16, 125)
    assert (8, 50) in fallbacks
    assert (6, 31) in fallbacks
    assert (4, 16) in fallbacks


def test_create_pod_with_fallbacks_retries_smaller_shape(monkeypatch):
    client = RunpodClient(api_key="rp_test")
    attempts = []

    def fake_create_pod(payload):
        attempts.append((payload["gpuTypeIds"][0], payload["minVCPUPerGPU"], payload["minRAMPerGPU"]))
        if len(attempts) == 1:
            raise RuntimeError("There are no instances currently available")
        return {"id": "pod_123"}

    monkeypatch.setattr(client, "create_pod", fake_create_pod)

    created = _create_pod_with_fallbacks(
        client=client,
        shared=_shared_config(),
        pod=PodLaunchConfig(name="aoede-train", gpu_type_id="NVIDIA RTX A5000"),
        network_volume_id=None,
    )

    assert attempts[0] == ("NVIDIA RTX A5000", 16, 125)
    assert attempts[1] == ("NVIDIA RTX A5000", 8, 50)
    assert created["selected_gpu_type_id"] == "NVIDIA RTX A5000"
    assert created["selected_vcpu_count"] == 8
    assert created["selected_memory_in_gb"] == 50
