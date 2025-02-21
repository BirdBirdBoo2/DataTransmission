import numpy as np


def _to_bits(num: int, n_bits: int) -> np.ndarray[np.uint8]:
    """
    Convert a number to a bit array.
    """
    num_as_bytes = int(num).to_bytes(2, byteorder='little')
    num_as_bytes = np.frombuffer(num_as_bytes, dtype=np.uint8)
    num_as_bits = np.unpackbits(num_as_bytes, bitorder='little')
    return num_as_bits[:n_bits]


def _from_bits(bits: np.ndarray[np.uint8]) -> int:
    """
    Convert a bit array to a number.
    """
    bits = np.packbits(bits, bitorder='little')
    return int.from_bytes(bits.tobytes(), byteorder='little')
