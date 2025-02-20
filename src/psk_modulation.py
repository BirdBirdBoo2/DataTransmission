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


class PSK_Modulation(Modulation):
    """
    This class implements M-ary Phase Shift Keying (M-PSK) modulation and demodulation.
    """
    def __init__(self, M: int):
        if M not in [2, 4, 8, 16]:
            raise ValueError("M-PSK is only implemented for M = 2, 4, 8, or 16")
        self.M = M
        self.symbol_map = self.generate_symbol_map()

    def generate_symbol_map(self):
        """
        Generate the PSK constellation points.
        """
        return {i: np.exp(1j * 2 * np.pi * i / self.M) for i in range(self.M)}

    def modulate(self, bits: np.ndarray) -> np.ndarray[np.complex128]:
        """
        Convert bit sequence into M-PSK symbols.
        """
        bits_per_symbol = int(np.log2(self.M))
        bit_groups = bits.reshape(-1, bits_per_symbol)
        bit_indices = np.packbits(bit_groups, axis=1, bitorder='little').flatten()
        symbols = np.array([self.symbol_map[i] for i in bit_indices])
        return symbols.astype(np.complex128)

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """
        Convert received symbols back into bits.
        """
        bits_per_symbol = int(np.log2(self.M))
        detected_indices = np.array([np.argmin(np.abs(symbol - np.array(list(self.symbol_map.values())))) for symbol in symbols])
        detected_bits = np.unpackbits(detected_indices[:, np.newaxis].astype(np.uint8), axis=1, bitorder='little')[:, -bits_per_symbol:]
        return detected_bits.flatten()
