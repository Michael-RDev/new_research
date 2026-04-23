from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class ProfileDefaults:
    output_root: str
    benchmark_output_dir: str
    max_train_examples: int
    max_eval_examples: int
    max_samples: int
    max_steps: int
    batch_size: int
    run_evals: bool
    prepare_eval_assets: bool


PROFILE_DEFAULTS = {
    "smoke": ProfileDefaults(
        output_root="exp/aoede_smoke",
        benchmark_output_dir="exp/benchmark_compare_aoede_smoke",
        max_train_examples=128,
        max_eval_examples=16,
        max_samples=256,
        max_steps=500,
        batch_size=2,
        run_evals=True,
        prepare_eval_assets=True,
    ),
    "core": ProfileDefaults(
        output_root="exp/aoede_stage1_core",
        benchmark_output_dir="exp/benchmark_compare_aoede_stage1_core",
        max_train_examples=1024,
        max_eval_examples=64,
        max_samples=0,
        max_steps=20_000,
        batch_size=4,
        run_evals=False,
        prepare_eval_assets=False,
    ),
    "full": ProfileDefaults(
        output_root="exp/aoede_mosaicflow_full",
        benchmark_output_dir="exp/benchmark_compare_aoede_mosaicflow_full",
        max_train_examples=0,
        max_eval_examples=256,
        max_samples=0,
        max_steps=100_000,
        batch_size=8,
        run_evals=False,
        prepare_eval_assets=False,
    ),
}


@dataclass(frozen=True)
class RunPodPaths:
    workspace: Path
    repo_root: Path
    omnivoice_dir: Path
    cloneval_repo: Path
    python_bin: Path
    manifest_path: Path
    env_file: str


@dataclass(frozen=True)
class ResolvedPipelineConfig:
    profile: str
    output_root: Path
    benchmark_output_dir: Path
    max_train_examples: int
    max_eval_examples: int
    max_samples: int
    max_steps: int
    batch_size: int
    run_evals: bool
    prepare_eval_assets: bool
    architecture_variant: str
    init_from_omnivoice: str
    learning_rate: Optional[float]
    checkpoint_every: Optional[int]
    seed: Optional[int]
    device: Optional[str]
    cloneval_test_list: Optional[Path]
    benchmarks: Optional[list[str]]

    @property
    def checkpoint_path(self) -> Path:
        return self.output_root / "artifacts" / "checkpoints" / "checkpoint-last.pt"


@dataclass(frozen=True)
class PipelinePlan:
    paths: RunPodPaths
    config: ResolvedPipelineConfig
    stage_command: list[str]
    train_command: list[str]
    audio_preflight_command: Optional[list[str]]
    eval_command: Optional[list[str]]
    core_handoff_command: Optional[list[str]]


def _summary_path(paths: RunPodPaths) -> Path:
    return paths.repo_root / "artifacts" / "manifests" / "atlasflow_hf_summary.json"


def _stage_completed_despite_abort(paths: RunPodPaths) -> bool:
    summary_path = _summary_path(paths)
    if not summary_path.exists() or not paths.manifest_path.exists():
        return False

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    return int(summary.get("train_entries", 0)) > 0


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RunPod Aoede pipeline for staged smoke/core training and optional evaluation.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_DEFAULTS.keys()),
        default="smoke",
        help="Training profile to run.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(os.environ.get("WORKSPACE", "/workspace")),
        help="RunPod workspace root.",
    )
    parser.add_argument(
        "--root-repo-dir-name",
        type=str,
        default=os.environ.get("ROOT_REPO_DIR_NAME", "new_research"),
        help="Repo directory name under the workspace when --root-repo-dir is omitted.",
    )
    parser.add_argument(
        "--root-repo-dir",
        type=Path,
        default=Path(os.environ["ROOT_REPO_DIR"]) if "ROOT_REPO_DIR" in os.environ else None,
        help="Absolute repo root. Defaults to $WORKSPACE/$ROOT_REPO_DIR_NAME.",
    )
    parser.add_argument(
        "--omnivoice-dir",
        type=Path,
        default=Path(os.environ["OMNIVOICE_DIR"]) if "OMNIVOICE_DIR" in os.environ else None,
        help="Absolute OmniVoice checkout path. Defaults to <repo>/OmniVoice.",
    )
    parser.add_argument(
        "--cloneval-repo",
        type=Path,
        default=Path(os.environ["CLONEVAL_REPO"]) if "CLONEVAL_REPO" in os.environ else None,
        help="Absolute CloneEval checkout path. Defaults to $WORKSPACE/cloneval.",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(os.environ["PYTHON_BIN"]) if "PYTHON_BIN" in os.environ else None,
        help="Python executable inside the RunPod workspace env.",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=os.environ.get("AOEDE_ENV_FILE", ".env"),
        help="Optional env file path passed to Hugging Face staging.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("artifacts/manifests/train.jsonl"),
        help="Train manifest path consumed by Aoede training.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the training output root for the selected profile.",
    )
    parser.add_argument(
        "--benchmark-output-dir",
        type=Path,
        default=None,
        help="Override the benchmark comparison output directory.",
    )
    parser.add_argument(
        "--run-evals",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run benchmark comparison after training. Defaults to on for smoke and off for core.",
    )
    parser.add_argument(
        "--prepare-eval-assets",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Download benchmark assets before evaluation. Defaults to on when evals are enabled.",
    )
    parser.add_argument(
        "--cloneval-test-list",
        type=Path,
        default=None,
        help="Optional CloneEval JSONL passed through to benchmark comparison.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Optional benchmark subset passed through to benchmark comparison.",
    )
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=None,
        help="Per-source train staging cap override.",
    )
    parser.add_argument(
        "--max-eval-examples",
        type=int,
        default=None,
        help="Per-source eval staging cap override.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Aoede train manifest subset cap override.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Aoede max training steps override.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Aoede training batch size override.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Aoede learning rate override.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Aoede checkpoint interval override.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Aoede random seed override.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Explicit Aoede training device.",
    )
    parser.add_argument(
        "--architecture-variant",
        type=str,
        default="mosaicflow",
        choices=["baseline", "atlasflow", "mosaicflow"],
        help="Aoede architecture variant.",
    )
    parser.add_argument(
        "--init-from-omnivoice",
        type=str,
        default=os.environ.get("OMNIVOICE_INIT", "k2-fsa/OmniVoice"),
        help="Warm-start source for Aoede initialization.",
    )
    return parser


def _resolve_path(base_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _absolute_path(base_dir: Path, path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return (base_dir / expanded).absolute()


def resolve_paths(args: argparse.Namespace) -> RunPodPaths:
    workspace = args.workspace.expanduser().resolve()
    repo_root = (
        args.root_repo_dir.expanduser().resolve()
        if args.root_repo_dir is not None
        else (workspace / args.root_repo_dir_name).resolve()
    )
    omnivoice_dir = (
        args.omnivoice_dir.expanduser().resolve()
        if args.omnivoice_dir is not None
        else (repo_root / "OmniVoice").resolve()
    )
    cloneval_repo = (
        args.cloneval_repo.expanduser().resolve()
        if args.cloneval_repo is not None
        else (workspace / "cloneval").resolve()
    )
    python_bin = (
        _absolute_path(repo_root, args.python_bin)
        if args.python_bin is not None
        else (repo_root / ".venv" / "bin" / "python").absolute()
    )
    manifest_path = _resolve_path(repo_root, args.manifest_path)
    return RunPodPaths(
        workspace=workspace,
        repo_root=repo_root,
        omnivoice_dir=omnivoice_dir,
        cloneval_repo=cloneval_repo,
        python_bin=python_bin,
        manifest_path=manifest_path,
        env_file=args.env_file,
    )


def resolve_pipeline_config(
    args: argparse.Namespace,
    paths: RunPodPaths,
) -> ResolvedPipelineConfig:
    profile_defaults = PROFILE_DEFAULTS[args.profile]
    output_root = _resolve_path(
        paths.workspace,
        args.output_root if args.output_root is not None else Path(profile_defaults.output_root),
    )
    benchmark_output_dir = _resolve_path(
        paths.workspace,
        args.benchmark_output_dir
        if args.benchmark_output_dir is not None
        else Path(profile_defaults.benchmark_output_dir),
    )
    run_evals = profile_defaults.run_evals if args.run_evals is None else args.run_evals
    prepare_eval_assets = (
        profile_defaults.prepare_eval_assets if args.prepare_eval_assets is None else args.prepare_eval_assets
    )
    return ResolvedPipelineConfig(
        profile=args.profile,
        output_root=output_root,
        benchmark_output_dir=benchmark_output_dir,
        max_train_examples=(
            profile_defaults.max_train_examples
            if args.max_train_examples is None
            else args.max_train_examples
        ),
        max_eval_examples=(
            profile_defaults.max_eval_examples
            if args.max_eval_examples is None
            else args.max_eval_examples
        ),
        max_samples=profile_defaults.max_samples if args.max_samples is None else args.max_samples,
        max_steps=profile_defaults.max_steps if args.max_steps is None else args.max_steps,
        batch_size=profile_defaults.batch_size if args.batch_size is None else args.batch_size,
        run_evals=run_evals,
        prepare_eval_assets=prepare_eval_assets,
        architecture_variant=args.architecture_variant,
        init_from_omnivoice=args.init_from_omnivoice,
        learning_rate=args.learning_rate,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        device=args.device,
        cloneval_test_list=(
            args.cloneval_test_list.expanduser().resolve()
            if args.cloneval_test_list is not None
            else None
        ),
        benchmarks=args.benchmarks,
    )


def build_stage_command(paths: RunPodPaths, config: ResolvedPipelineConfig) -> list[str]:
    return [
        str(paths.python_bin),
        "-m",
        "aoede.data.huggingface",
        "--project-root",
        str(paths.repo_root),
        "--env-file",
        paths.env_file,
        "--max-train-examples",
        str(config.max_train_examples),
        "--max-eval-examples",
        str(config.max_eval_examples),
    ]


def build_train_command(paths: RunPodPaths, config: ResolvedPipelineConfig) -> list[str]:
    command = [
        str(paths.python_bin),
        "-m",
        "aoede.training.train_aoede",
        "--source-manifest",
        str(paths.manifest_path),
        "--output-root",
        str(config.output_root),
        "--architecture-variant",
        config.architecture_variant,
        "--batch-size",
        str(config.batch_size),
        "--max-steps",
        str(config.max_steps),
        "--max-samples",
        str(config.max_samples),
        "--init-from-omnivoice",
        config.init_from_omnivoice,
    ]
    if config.learning_rate is not None:
        command.extend(["--learning-rate", str(config.learning_rate)])
    if config.checkpoint_every is not None:
        command.extend(["--checkpoint-every", str(config.checkpoint_every)])
    if config.seed is not None:
        command.extend(["--seed", str(config.seed)])
    if config.device is not None:
        command.extend(["--device", config.device])
    return command


def build_audio_preflight_command(paths: RunPodPaths, config: ResolvedPipelineConfig) -> Optional[list[str]]:
    if not config.run_evals:
        return None
    return ["bash", str(paths.repo_root / "scripts" / "runpod" / "check_audio_stack.sh")]


def build_eval_command(paths: RunPodPaths, config: ResolvedPipelineConfig) -> Optional[list[str]]:
    if not config.run_evals:
        return None

    command = [
        str(paths.python_bin),
        "-m",
        "omnivoice.scripts.benchmark_compare",
        "--baseline-model",
        "k2-fsa/OmniVoice",
        "--candidate-model",
        str(config.checkpoint_path),
        "--candidate-label",
        "Aoede-MosaicFlow",
        "--candidate-infer-module",
        "aoede.eval.infer_batch",
        "--candidate-cloneval-module",
        "aoede.eval.cloneval_benchmark",
        "--output-dir",
        str(config.benchmark_output_dir),
        "--cloneval-repo",
        str(paths.cloneval_repo),
    ]
    if config.prepare_eval_assets:
        command.append("--prepare-assets")
    if config.cloneval_test_list is not None:
        command.extend(["--cloneval-test-list", str(config.cloneval_test_list)])
    if config.benchmarks:
        command.extend(["--benchmarks", *config.benchmarks])
    return command


def build_core_handoff_command(paths: RunPodPaths, config: ResolvedPipelineConfig) -> Optional[list[str]]:
    if config.profile != "smoke":
        return None

    command = [
        "bash",
        str(paths.repo_root / "scripts" / "runpod" / "run_aoede_pipeline.sh"),
        "--profile",
        "core",
        "--architecture-variant",
        config.architecture_variant,
        "--init-from-omnivoice",
        config.init_from_omnivoice,
    ]
    if config.learning_rate is not None:
        command.extend(["--learning-rate", str(config.learning_rate)])
    if config.checkpoint_every is not None:
        command.extend(["--checkpoint-every", str(config.checkpoint_every)])
    if config.seed is not None:
        command.extend(["--seed", str(config.seed)])
    if config.device is not None:
        command.extend(["--device", config.device])
    return command


def build_pipeline_plan(args: argparse.Namespace) -> PipelinePlan:
    paths = resolve_paths(args)
    config = resolve_pipeline_config(args, paths)
    return PipelinePlan(
        paths=paths,
        config=config,
        stage_command=build_stage_command(paths, config),
        train_command=build_train_command(paths, config),
        audio_preflight_command=build_audio_preflight_command(paths, config),
        eval_command=build_eval_command(paths, config),
        core_handoff_command=build_core_handoff_command(paths, config),
    )


def _run_command(command: Sequence[str], *, cwd: Path) -> None:
    subprocess.run(list(command), cwd=str(cwd), check=True)


def execute_pipeline(plan: PipelinePlan) -> None:
    print(f"Stage training data ({plan.config.profile})", flush=True)
    try:
        _run_command(plan.stage_command, cwd=plan.paths.repo_root)
    except subprocess.CalledProcessError as exc:
        if exc.returncode in {-6, 134} and _stage_completed_despite_abort(plan.paths):
            print(
                "Stage subprocess aborted during interpreter shutdown after writing manifests; "
                "reusing staged training data and continuing.",
                flush=True,
            )
        else:
            raise

    print(f"Train Aoede ({plan.config.profile})", flush=True)
    _run_command(plan.train_command, cwd=plan.paths.repo_root)

    if plan.eval_command is not None:
        print("Verify audio stack before benchmark evaluation", flush=True)
        if plan.audio_preflight_command is not None:
            _run_command(plan.audio_preflight_command, cwd=plan.paths.repo_root)

        print("Run benchmark comparison and CloneEval", flush=True)
        _run_command(plan.eval_command, cwd=plan.paths.omnivoice_dir)

    if plan.core_handoff_command is not None:
        print("Full/core training command:", flush=True)
        print(shlex.join(plan.core_handoff_command), flush=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = get_parser().parse_args(argv)
    execute_pipeline(build_pipeline_plan(args))


if __name__ == "__main__":
    main()
