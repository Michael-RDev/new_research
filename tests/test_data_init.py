import importlib
import sys


def test_aoede_data_import_is_lazy_for_torch_backed_dataset_module():
    sys.modules.pop("aoede.data", None)
    sys.modules.pop("aoede.data.dataset", None)

    data_module = importlib.import_module("aoede.data")

    assert "aoede.data.dataset" not in sys.modules
    assert callable(data_module.atlasflow_default_requests)
