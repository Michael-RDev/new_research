from pathlib import Path

from aoede.training.train_aoede import get_parser


def test_train_aoede_parser_accepts_resume_from():
    args = get_parser().parse_args(["--resume-from", "artifacts/checkpoints/checkpoint-last.pt"])

    assert args.resume_from == Path("artifacts/checkpoints/checkpoint-last.pt")
