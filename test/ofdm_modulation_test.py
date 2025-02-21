import pytest
import numpy as np

from src.encoder import Encoder

testdata = np.frombuffer(b'hello, world!', dtype=np.uint8)

@pytest.fixture
def ofdm_encoder():
    encoder = Encoder(n_subcarriers=64, cyclic_prefix_length=16, modulation='bpsk')
    return encoder


def test_encode(ofdm_encoder: Encoder):
    signal = ofdm_encoder.encode(bytearray(b'hello, world!'))

    assert len(signal) > 0

