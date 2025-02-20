import pytest
import numpy as np

from src.psk_modulation import BPSK_Modulation, Modulation, QPSK_Modulation

testdata = np.array(bytearray(b'hello, world!'))

@pytest.fixture
def bpsk():
    return BPSK_Modulation(16)

@pytest.fixture
def qpsk():
    return QPSK_Modulation()

def test_modulate_bpsk(bpsk: Modulation):
    bits = np.unpackbits(testdata)

    symbols = bpsk.modulate(bits)
    assert symbols.dtype == np.complex128


def test_modulate_demodulate_bpsk(bpsk: Modulation):
    bits = np.unpackbits(testdata)

    symbols = bpsk.modulate(bits)
    assert symbols.dtype == np.complex128

    recovered_bits = bpsk.demodulate(symbols)
    assert len(recovered_bits) == len(bits)
    assert np.allclose(recovered_bits, bits)


def test_modulate_qpsk(qpsk: Modulation):
    bits = np.unpackbits(testdata)

    symbols = qpsk.modulate(bits)
    assert symbols.dtype == np.complex128
    assert len(symbols) == len(bits) // 2
