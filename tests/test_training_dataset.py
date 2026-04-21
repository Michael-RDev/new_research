import torch

from aoede.data.dataset import TrainingExample, collate_training_examples
from aoede.languages import language_index


def _example(language_code: str) -> TrainingExample:
    return TrainingExample(
        text="sample",
        language_code=language_code,
        token_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        waveform=torch.zeros(8, dtype=torch.float32),
        codec_latents=torch.zeros(2, 4, dtype=torch.float32),
        reference_latents=torch.zeros(1, 4, dtype=torch.float32),
        reference_mask=torch.ones(1, dtype=torch.bool),
        prosody_targets=torch.zeros(1, 6, dtype=torch.float32),
        durations=torch.ones(3, dtype=torch.long),
        speaker_ref=torch.zeros(5, dtype=torch.float32),
        has_reference=False,
    )


def test_collate_training_examples_assigns_nonzero_ids_to_experimental_and_named_languages():
    batch = collate_training_examples(
        [
            _example("wo"),
            _example("english"),
            _example("mandarin chinese"),
        ]
    )

    assert batch["language_ids"].tolist() == [
        language_index("wo"),
        language_index("english"),
        language_index("mandarin chinese"),
    ]
