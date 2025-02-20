from abc import ABC, abstractmethod
import numpy as np


class Modulation(ABC):
    @abstractmethod
    def modulate(self, bits: np.ndarray) -> np.ndarray[np.complex128]:
        pass

    @abstractmethod
    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        pass


class BPSK_Modulation(Modulation):
    """
    This class implements the modulation of bits into Binary Shift Keying symbols.
    """
    def __init__(self):
        self.bpsk_mapping = {
            0: 1,
            1: -1,
        }

    def modulate(self, bits):
        symbols = np.array([self.bpsk_mapping[b] for b in bits])
        return symbols.astype(np.complex128)

    def demodulate(self, symbols):
        pass


class QPSK_Modulation(Modulation):
    """
    This class implements the modulation of bits into Quadratic Shift Keying symbols.
    """
    def __init__(self):
        self.qpsk_mapping = {
            (0, 0): 1 + 1j,
            (0, 1): -1 + 1j,
            (1, 0): 1 - 1j,
            (1, 1): -1 - 1j,
        }

        self.qpsk_direct_mapping = [v for _, v in self.qpsk_mapping.items()]


    def modulate(self, bits):
        bit_pairs = bits.reshape(-1, 2)
        bit_indices = np.packbits(bit_pairs, axis=1, bitorder='little').flatten()
        symbols = np.array([self.qpsk_direct_mapping[b] for b in bit_indices])
        return symbols.astype(np.complex128)

    def demodulate(self, symbols):
        pass