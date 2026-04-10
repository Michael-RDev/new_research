import json
from pathlib import Path

from omnivoice.scripts.orchestrate_tokenization import _build_data_config


def test_build_data_config_applies_core_repeat_weighting(tmp_path):
    token_root = tmp_path / "tokens"
    token_root.mkdir()

    waxal_dir = token_root / "waxal_am_train"
    waxal_dir.mkdir()
    (waxal_dir / "data.lst").write_text(
        f"{waxal_dir}/audios/shard-000000.tar {waxal_dir}/txts/shard-000000.jsonl 10 1800\n",
        encoding="utf-8",
    )
    (waxal_dir / "manifest.summary.json").write_text(
        json.dumps(
            {
                "dataset_name": "waxal",
                "split_role": "train",
                "language_id": "am",
                "repeat_hint": 4,
            }
        ),
        encoding="utf-8",
    )

    dev_dir = token_root / "emilia_en_dev"
    dev_dir.mkdir()
    (dev_dir / "data.lst").write_text(
        f"{dev_dir}/audios/shard-000000.tar {dev_dir}/txts/shard-000000.jsonl 10 3600\n",
        encoding="utf-8",
    )
    (dev_dir / "manifest.summary.json").write_text(
        json.dumps(
            {
                "dataset_name": "emilia",
                "split_role": "dev",
                "language_id": "en",
                "repeat_hint": 1,
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "data_config.json"
    _build_data_config(token_root, output_path, profile="core")
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["train"][0]["repeat"] == 8
    assert payload["train"][0]["dataset_name"] == "waxal"
    assert payload["dev"][0]["repeat"] == 1

