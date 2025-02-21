from ctypes import cast

import numpy as np
from pathlib import Path
from scipy.io import wavfile as wav

from src.psk_modulation import Modulation, QPSK_Modulation, BPSK_Modulation

DEFAULT_SAMPLE_RATE = 44100


class Decoder:
    def __init__(self, n_subcarriers: int = 64, cyclic_prefix_length: int = 16, modulation: str = 'qpsk'):
        self.n_subcarriers = n_subcarriers  # Number of subcarriers
        self.cyclic_prefix_length = cyclic_prefix_length  # Cyclic prefix length

        self.modulation: Modulation
        if modulation == 'qpsk':
            self.modulation = QPSK_Modulation()
        elif modulation == 'bpsk':
            self.modulation = BPSK_Modulation()
        else:
            raise ValueError(f"Invalid modulation type: {modulation}")

    def decode_file(self, file_path: Path) -> np.ndarray[np.uint8]:
        sample_rate, wave_signal = wav.read(file_path)
        return self.decode(wave_signal)

    def decode(self, wave_signal: np.ndarray[np.uint16]) -> np.ndarray[np.uint8]:
        num_sub = self.n_subcarriers
        cyclic_prefix_length = self.cyclic_prefix_length

        time_domain_symbols = wave_signal.reshape(-1, num_sub + cyclic_prefix_length)
        time_domain_symbols = time_domain_symbols[:, cyclic_prefix_length:]

        # Perform FFT
        ofdm_symbols = np.fft.fft(time_domain_symbols, axis=1)

        # Extract useful subcarriers

        # noinspection PyTypeChecker
        symbols: np.ndarray[np.complex128] = ofdm_symbols[:, 1:num_sub // 2].flatten().astype(np.complex128)
        return self.modulation.demodulate(symbols)

if __name__ == "__main__":
    decoder = Decoder(modulation='bpsk')
    decoded_data = decoder.decode_file(Path("../text.txt.wav"))