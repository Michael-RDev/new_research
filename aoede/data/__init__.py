from importlib import import_module

from aoede.data.alignments import proportional_durations
from aoede.data.huggingface import (
    HFIngestRequest,
    atlasflow_default_requests,
    prepare_atlasflow_training_assets,
    prepare_hf_source,
    supported_hf_datasets,
)
from aoede.data.manifest import CORPUS_PLAN, ManifestEntry, load_manifest, save_manifest

_LAZY_EXPORTS = {
    "ManifestDataset": ("aoede.data.dataset", "ManifestDataset"),
    "TrainingExample": ("aoede.data.dataset", "TrainingExample"),
    "collate_training_examples": ("aoede.data.dataset", "collate_training_examples"),
}

__all__ = [
    "CORPUS_PLAN",
    "HFIngestRequest",
    "ManifestDataset",
    "ManifestEntry",
    "TrainingExample",
    "atlasflow_default_requests",
    "collate_training_examples",
    "load_manifest",
    "prepare_atlasflow_training_assets",
    "prepare_hf_source",
    "proportional_durations",
    "save_manifest",
    "supported_hf_datasets",
]


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attribute_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attribute_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
