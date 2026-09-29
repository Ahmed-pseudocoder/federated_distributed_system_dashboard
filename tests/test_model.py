import torch

from core.model import create_model


def test_model_output_shape():
    output = create_model()(torch.zeros(4, 1, 28, 28))
    assert output.shape == (4, 10)
