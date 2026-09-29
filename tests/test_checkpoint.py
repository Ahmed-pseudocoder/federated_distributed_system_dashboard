import torch

from coordinator.checkpoint import CheckpointManager
from core.model import create_model


def test_checkpoint_save_and_restore(tmp_path):
    manager = CheckpointManager(tmp_path)
    model = create_model()
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    path = manager.save(3, model, {"accuracy": 0.5})
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    restored_path, checkpoint = manager.restore(model)
    assert restored_path == path
    assert checkpoint["round"] == 3
    assert all(torch.equal(model.state_dict()[key], value) for key, value in expected.items())
