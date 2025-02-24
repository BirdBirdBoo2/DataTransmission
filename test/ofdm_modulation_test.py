import pytest
import numpy as np

from src import Decoder, Encoder

testdata = np.frombuffer(b'hello, world!', dtype=np.uint8)

@pytest.fixture
def ofdm_encoder():
    encoder = Encoder(n_subcarriers=64,
                      cyclic_prefix_length=16,
                      modulation='bpsk',
                      ofdm_frequency_hz=512,
                      carrier_frequency_hz=2048,
                      grace_period_s=0)
    return encoder

@pytest.fixture
def ofdm_decoder():
    decoder = Decoder(n_subcarriers=64,
                      cyclic_prefix_length=16,
                      modulation='bpsk',
                      ofdm_frequency_hz=512,
                      carrier_frequency_hz=2048,
                      grace_period_s=0)
    return decoder


def test_encode(ofdm_encoder: Encoder):
    signal = ofdm_encoder.encode(bytearray(b'hello, world!'))

    assert len(signal) > 0

def test_decoder(ofdm_encoder: Encoder, ofdm_decoder: Decoder):
    signal = ofdm_encoder.encode(bytearray(b'hello, world!'))
    decoded = ofdm_decoder.decode(signal)

    np.testing.assert_array_equal(decoded, testdata)

