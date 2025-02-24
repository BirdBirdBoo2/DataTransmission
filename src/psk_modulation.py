from abc import ABC, abstractmethod
from typing import cast

import structlog

from .bitmagic import _to_bits, _from_bits

import numpy as np


def _chunk_array(array: np.ndarray, chunk_size: int) -> tuple[np.ndarray, int]:
    """
    Split an array into chunks of a given size.
    :returns: Chunked array and the padding length.
    """
    pad_length = len(array) % chunk_size
    pad_length = (chunk_size - pad_length) if pad_length != 0 else 0

    array = np.concatenate([array, np.zeros(pad_length, dtype=array.dtype)])
    return array.reshape(-1, chunk_size), pad_length



def _make_headers(n_chunks: int, prefix_length_bits: int, max_chunk_size_bits: int,
                  last_chunk_padding_bits: int) -> np.ndarray:
    """
    Generate the headers for each data chunk.
    :return: N x prefix_length_bits array of headers, where N is the number of chunks.
    """
    max_data_chunk_as_bits = _to_bits(max_chunk_size_bits, prefix_length_bits)
    last_data_chunk_size_as_bits = _to_bits(max_chunk_size_bits - last_chunk_padding_bits, prefix_length_bits)
    header_bits = np.repeat([max_data_chunk_as_bits], n_chunks, axis=0)
    header_bits[-1, :] = last_data_chunk_size_as_bits

    assert header_bits.shape[0] == n_chunks
    assert header_bits.shape[1] == prefix_length_bits

    return header_bits


def _recover_headers(header_bits: np.ndarray) -> np.ndarray:
    """
    Recover the headers from the received data.
    """
    return np.packbits(header_bits, axis=1, bitorder='big')


class Modulation(ABC):
    @abstractmethod
    def modulate(self, bits: np.ndarray) -> np.ndarray[np.complex128]:
        pass

    @abstractmethod
    def demodulate(self, symbols: np.ndarray[np.complex128]) -> np.ndarray[np.uint8]:
        pass


class BPSK_Modulation(Modulation):
    """
    This class implements the modulation of bits into Binary Shift Keying symbols.
    """

    def __init__(self, n_subcarriers: int = 64):
        self.bpsk_mapping = {
            0: 1,
            1: -1,
        }

        self.bpsk_reverse_mapping = {v: np.uint8(k) for k, v in self.bpsk_mapping.items()}

        self.n_subcarriers = n_subcarriers

        bits_per_symbol = 1  # in bpsk we encode 1 bit per symbol
        self.prefix_length_bits = np.ceil(np.log2(n_subcarriers * bits_per_symbol)).astype(int)

        # number of data bits in message chunk
        self.max_data_chunk_size_bits = (self.n_subcarriers * bits_per_symbol - self.prefix_length_bits)
        self.logger = structlog.getLogger(BPSK_Modulation.__name__)


    def modulate(self, bits):
        chunk_size = self.max_data_chunk_size_bits
        data_chunks, padding = _chunk_array(bits, chunk_size)
        n_chunks = data_chunks.shape[0]

        headers = _make_headers(n_chunks, self.prefix_length_bits, self.max_data_chunk_size_bits, padding)
        message_bits = np.hstack((headers, data_chunks)).flatten()

        self.logger.info(f'Encoded 32 first bits', message_bits=message_bits[:32])

        symbols = np.array([self.bpsk_mapping[b] for b in message_bits])

        return symbols.astype(np.complex128)

    def demodulate(self, symbols: np.ndarray[np.complex128]) -> np.ndarray[np.uint8]:
        symbols = np.round(symbols, 0)
        symbols = np.real(symbols)
        recovered_data_bits = np.array([self.bpsk_reverse_mapping[s] for s in symbols], dtype=np.uint8)
        self.logger.info(f'Recovered 32 first bits', message_bits=recovered_data_bits[:32])
        recovered_data_chunks = recovered_data_bits.reshape(-1, self.n_subcarriers)

        bits = []

        for i in range(0, recovered_data_chunks.shape[0]):
            chunk = recovered_data_chunks[i]
            chunk_length_bits = chunk[:self.prefix_length_bits]
            chunk_length = _from_bits(chunk_length_bits)
            bits.append(chunk[self.prefix_length_bits:][:chunk_length])

        return np.concatenate(bits)


class QPSK_Modulation(Modulation):
    """
    This class implements the modulation of bits into Quadratic Shift Keying symbols.
    """

    def __init__(self, n_subcarriers: int = 64):
        self.qpsk_mapping = {
            (0, 0): 1 + 1j,
            (0, 1): -1 + 1j,
            (1, 0): 1 - 1j,
            (1, 1): -1 - 1j,
        }

        self.qpsk_direct_mapping = [v for _, v in self.qpsk_mapping.items()]
        self.qpsk_reverse_mapping = {v: k for k, v in self.qpsk_mapping.items()}

        self.n_subcarriers = n_subcarriers
        bits_per_symbol = 2  # QPSK encodes 2 bits per symbol
        self.prefix_length_bits = np.ceil(np.log2(n_subcarriers * bits_per_symbol)).astype(int)
        self.max_data_chunk_size_bits = (self.n_subcarriers * bits_per_symbol - self.prefix_length_bits)
        self.logger = structlog.getLogger(QPSK_Modulation.__name__)

    def _closest_qpsk_symbol(self, symbol):
        return min(self.qpsk_reverse_mapping.keys(), key=lambda s: np.abs(s - symbol))

    def modulate(self, bits):
        chunk_size = self.max_data_chunk_size_bits
        data_chunks, padding = _chunk_array(bits, chunk_size)
        n_chunks = data_chunks.shape[0]

        headers = _make_headers(n_chunks, self.prefix_length_bits, self.max_data_chunk_size_bits, padding)
        message_bits = np.hstack((headers, data_chunks)).flatten()
        self.logger.info(f'Encoded 32 first bits', message_bits=message_bits[:32])

        bit_pairs = message_bits.reshape(-1, 2)
        symbols = np.array([self.qpsk_mapping[tuple(b)] for b in bit_pairs])

        return symbols.astype(np.complex128)

    def demodulate(self, symbols: np.ndarray[np.complex128]) -> np.ndarray[np.uint8]:
        symbols = np.real_if_close(symbols)

        normalized_symbols = np.array([self._closest_qpsk_symbol(s) for s in symbols])
        recovered_data_pairs = np.array([self.qpsk_reverse_mapping[s] for s in normalized_symbols], dtype=np.uint8)
        recovered_data_bits = recovered_data_pairs.flatten()
        
        self.logger.info(f'Recovered 32 first bits', message_bits=recovered_data_bits[:32])
        recovered_data_chunks = recovered_data_bits.reshape(-1, self.n_subcarriers * 2)

        bits = []
        for i in range(recovered_data_chunks.shape[0]):
            chunk = recovered_data_chunks[i]
            chunk_length_bits = chunk[:self.prefix_length_bits]
            chunk_length = _from_bits(chunk_length_bits)
            bits.append(chunk[self.prefix_length_bits:][:chunk_length])

        return np.concatenate(bits)


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
        detected_indices = np.array(
            [np.argmin(np.abs(symbol - np.array(list(self.symbol_map.values())))) for symbol in symbols])
        detected_bits = np.unpackbits(detected_indices[:, np.newaxis].astype(np.uint8), axis=1, bitorder='little')[:,
                        -bits_per_symbol:]
        return detected_bits.flatten()
