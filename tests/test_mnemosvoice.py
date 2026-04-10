import types

import torch
from transformers import BertConfig

from omnivoice.data.collator import PackingDataCollator
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig


class DummyLLM(torch.nn.Module):
    def __init__(self, vocab_size=64, hidden_size=32):
        super().__init__()
        self.config = types.SimpleNamespace(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size),
        )

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def resize_token_embeddings(self, new_size):
        self.embed = torch.nn.Embedding(new_size, self.config.hidden_size)

    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
        return_dict=True,
        position_ids=None,
    ):
        return (self.net(inputs_embeds),)


class DummyTokenizer:
    pad_token_id = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [((ord(ch) % 13) + 1) for ch in text] or [1]
        if return_tensors == "pt":
            return types.SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))
        return types.SimpleNamespace(input_ids=ids)


def build_model():
    llm_config = BertConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
    )
    config = OmniVoiceConfig(
        audio_vocab_size=17,
        audio_mask_id=16,
        num_audio_codebook=2,
        audio_codebook_weights=[1, 1],
        architecture_variant="mnemosvoice",
        speaker_memory_num_tokens=4,
        planner_stride=2,
        planner_loss_weight=0.2,
        speaker_loss_weight=0.1,
        memory_conditioning_num_heads=4,
        llm_config=llm_config,
    )
    model = OmniVoice(config=config, llm=DummyLLM(vocab_size=64, hidden_size=32))
    model.text_tokenizer = DummyTokenizer()
    model.to("cpu")
    return model


def test_mnemosvoice_forward_reports_aux_losses():
    model = build_model()

    input_ids = torch.tensor(
        [
            [
                [3, 4, 5, 6, 7, 16, 9, 16, 11],
                [3, 4, 2, 3, 4, 16, 5, 16, 6],
            ]
        ],
        dtype=torch.long,
    )
    audio_mask = torch.tensor([[False, False, True, True, True, True, True, True, True]])
    prompt_mask = torch.tensor([[False, False, True, True, True, False, False, False, False]])
    target_audio_mask = torch.tensor([[False, False, False, False, False, True, True, True, True]])
    labels = torch.full_like(input_ids, -100)
    labels[:, :, 5] = torch.tensor([[8, 4]])
    labels[:, :, 7] = torch.tensor([[10, 7]])

    outputs = model(
        input_ids=input_ids,
        audio_mask=audio_mask,
        prompt_mask=prompt_mask,
        target_audio_mask=target_audio_mask,
        labels=labels,
    )

    assert outputs.logits.shape == (1, 2, 9, 17)
    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)
    assert outputs.plan_loss is not None
    assert torch.isfinite(outputs.plan_loss)
    assert outputs.speaker_loss is not None
    assert torch.isfinite(outputs.speaker_loss)


def test_prepare_inference_inputs_marks_prompt_and_target_spans():
    model = build_model()
    ref_audio_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    prepared = model._prepare_inference_inputs(
        text="hello",
        num_target_tokens=4,
        ref_text="greetings",
        ref_audio_tokens=ref_audio_tokens,
        lang="en",
        instruct="bright",
        denoise=True,
    )

    assert prepared["input_ids"].shape[1] == 2
    assert prepared["prompt_mask"].sum().item() == 3
    assert prepared["target_audio_mask"].sum().item() == 4
    assert prepared["audio_mask"].sum().item() == 7


def test_packing_collator_preserves_prompt_and_target_masks():
    processor = types.SimpleNamespace(text_tokenizer=types.SimpleNamespace(pad_token_id=0))
    collator = PackingDataCollator(processor, batch_tokens=10)
    samples = [
        {
            "input_ids": torch.ones(2, 4, dtype=torch.long),
            "labels": torch.full((2, 4), -100, dtype=torch.long),
            "audio_mask": torch.tensor([False, True, True, True]),
            "prompt_mask": torch.tensor([False, True, False, False]),
            "target_audio_mask": torch.tensor([False, False, True, True]),
            "length": 4,
        },
        {
            "input_ids": torch.ones(2, 3, dtype=torch.long) * 2,
            "labels": torch.full((2, 3), -100, dtype=torch.long),
            "audio_mask": torch.tensor([False, True, True]),
            "prompt_mask": torch.tensor([False, False, True]),
            "target_audio_mask": torch.tensor([False, True, False]),
            "length": 3,
        },
    ]

    batch = collator(samples)
    assert batch["prompt_mask"].shape == (1, 10)
    assert batch["target_audio_mask"].shape == (1, 10)
    assert batch["prompt_mask"][0, 1].item() is True
    assert batch["prompt_mask"][0, 6].item() is True
    assert batch["target_audio_mask"][0, 2].item() is True
    assert batch["target_audio_mask"][0, 5].item() is True
