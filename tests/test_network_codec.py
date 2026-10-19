from collections import OrderedDict

import torch

from distributed.codec import decode_state_dict, encode_state_dict


def test_network_state_dict_round_trip():
    source = OrderedDict(
        weight=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        counter=torch.tensor(7, dtype=torch.int64),
    )
    restored = decode_state_dict(encode_state_dict(source))
    assert list(restored) == list(source)
    assert all(torch.equal(restored[key], value) for key, value in source.items())
