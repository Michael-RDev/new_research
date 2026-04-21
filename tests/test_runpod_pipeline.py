from pathlib import Path

from aoede.runpod.pipeline import build_pipeline_plan, get_parser, resolve_paths


def _parse(*args: str):
    return get_parser().parse_args(
        [
            "--workspace",
            "/workspace",
            "--root-repo-dir",
            "/workspace/new_research",
            "--python-bin",
            "/workspace/new_research/.venv/bin/python",
            *args,
        ]
    )


def test_smoke_pipeline_builds_stage_train_eval_and_core_handoff():
    plan = build_pipeline_plan(
        _parse("--cloneval-test-list", "/workspace/data/cloneval/cloneval_smoke.jsonl")
    )

    assert plan.config.profile == "smoke"
    assert plan.config.max_train_examples == 128
    assert plan.config.max_eval_examples == 16
    assert plan.config.run_evals is True
    assert plan.stage_command[-4:] == ["--max-train-examples", "128", "--max-eval-examples", "16"]
    assert str(plan.config.output_root) == "/workspace/exp/aoede_smoke"
    assert "--prepare-assets" in plan.eval_command
    assert "--cloneval-test-list" in plan.eval_command
    assert str(plan.config.checkpoint_path) in plan.eval_command
    assert plan.core_handoff_command[:4] == [
        "bash",
        "/workspace/new_research/scripts/runpod/run_aoede_pipeline.sh",
        "--profile",
        "core",
    ]


def test_core_pipeline_reuses_single_entrypoint_and_skips_evals_by_default():
    plan = build_pipeline_plan(_parse("--profile", "core"))

    assert plan.config.profile == "core"
    assert plan.config.max_train_examples == 1024
    assert plan.config.max_eval_examples == 64
    assert plan.config.run_evals is False
    assert str(plan.config.output_root) == "/workspace/exp/aoede_stage1_core"
    assert plan.eval_command is None
    assert plan.audio_preflight_command is None
    assert plan.core_handoff_command is None
    assert plan.train_command[0:3] == [
        "/workspace/new_research/.venv/bin/python",
        "-m",
        "aoede.training.train_aoede",
    ]


def test_resolve_paths_keeps_virtualenv_python_path_instead_of_resolving_symlink(tmp_path: Path):
    workspace = tmp_path / "workspace"
    repo_root = workspace / "new_research"
    venv_bin = repo_root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    python_link = venv_bin / "python"
    python_link.symlink_to("/usr/bin/python3")

    args = get_parser().parse_args(
        [
            "--workspace",
            str(workspace),
            "--root-repo-dir",
            str(repo_root),
            "--python-bin",
            str(python_link),
        ]
    )

    paths = resolve_paths(args)

    assert paths.python_bin == python_link
