import importlib.util
from pathlib import Path

import torch


def load_trainer():
    path = Path(__file__).parents[1] / "code" / "20_train_anchor_tree_mlp_noreplay.py"
    spec = importlib.util.spec_from_file_location("parcil_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_support_centers_are_normalized_and_deterministic():
    trainer = load_trainer()
    generator = torch.Generator().manual_seed(11)
    features = torch.randn(100, 32, generator=generator)
    first = trainer.support_centers(features, 8)
    second = trainer.support_centers(features, 8)
    assert first.shape == (8, 32)
    torch.testing.assert_close(first.norm(dim=1), torch.ones(8), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(first, second)

