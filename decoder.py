import numpy as np
from pathlib import Path
from scipy.io import wavfile as wav

DEFAULT_SAMPLE_RATE = 44100


class Decoder:
    def __init__(self, n_subcarriers: int = 64, cyclic_prefix_length: int = 16):
        self.n_subcarriers = n_subcarriers  # Number of subcarriers
        self.cyclic_prefix_length = cyclic_prefix_length  # Cyclic prefix length
        self.qpsk_mapping = {
            (0, 0): 1 + 1j,
            (0, 1): -1 + 1j,
            (1, 0): 1 - 1j,
            (1, 1): -1 - 1j,
        }