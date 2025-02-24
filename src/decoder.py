from ctypes import cast

import numpy as np
from pathlib import Path

import structlog
from scipy.io import wavfile as wav

from src.psk_modulation import Modulation, QPSK_Modulation, BPSK_Modulation, PSK_Modulation

logger = structlog.stdlib.get_logger(__name__)


class Decoder:
    def __init__(self,
                 n_subcarriers: int = 64,
                 cyclic_prefix_length: int = 16,
                 modulation: str = 'qpsk',
                 M: int = 4,
                 grace_period_s: float = 0.01,
                 ofdm_frequency_hz: float = 512,
                 carrier_frequency_hz: float = 2048):
        self.ofdm_frequency_hz = ofdm_frequency_hz
        self.carrier_frequency_hz = carrier_frequency_hz
        self.sample_rate = int(4 * carrier_frequency_hz)

        self.n_subcarriers = n_subcarriers  # Number of subcarriers
        self.cyclic_prefix_length = cyclic_prefix_length  # Cyclic prefix length
        self.grace_period_s = grace_period_s

        self.modulation: Modulation
        if modulation == 'qpsk':
            self.modulation = QPSK_Modulation(n_subcarriers)
        elif modulation == 'bpsk':
            self.modulation = BPSK_Modulation(n_subcarriers)
        elif modulation == 'mpsk':
            self.modulation = PSK_Modulation(M)
        else:
            raise ValueError(f"Invalid modulation type: {modulation}")

    def _estimate_symbols(self, wave_signal: np.ndarray) -> np.ndarray:
        # TODO: improve
        estimated_num_pts_per_symbol = self.ofdm_frequency_hz * 2
        logger.info("Estimate symbols", num_pts_per_symbol=estimated_num_pts_per_symbol)
        return wave_signal.reshape((-1, estimated_num_pts_per_symbol))

    def _remove_carrier(self, signal: np.ndarray) -> np.ndarray:
        t = np.arange(0, signal.shape[1] / self.sample_rate, 1 / self.sample_rate)
        carrier_signal = np.sin(2 * np.pi * t * self.carrier_frequency_hz)
        signal = signal * carrier_signal

        fft = np.fft.fft(signal, axis=1)
        freq = np.fft.fftfreq(signal.shape[1], 1 / self.sample_rate)
        freq_mask = np.abs(freq) > self.ofdm_frequency_hz
        fft[:, freq_mask] = 0
        return fft

    def _get_ofdm_subcarriers(self, fft_coeffs: np.ndarray) -> np.ndarray:
        subc_indices = np.arange(1, self.n_subcarriers + 1)
        coeffs = fft_coeffs[:, subc_indices]
        coeffs_normed = coeffs/ np.abs(coeffs)
        return coeffs_normed


    def decode_file(self, file_path: Path) -> np.ndarray[np.uint8]:
        sample_rate, wave_signal = wav.read(file_path)

        assert sample_rate == self.sample_rate, f"Sample rate mismatch: {sample_rate} != {self.sample_rate}"

        wave_signal = wave_signal.astype(np.float64)
        wave_signal = wave_signal / np.max(np.abs(wave_signal))
        return self.decode(wave_signal)

    def decode(self, wave_signal: np.ndarray[float]) -> np.ndarray[np.uint8]:
        signal_modulated_symbols = self._estimate_symbols(wave_signal)
        modulated_symbols = self._remove_carrier(signal_modulated_symbols)
        psk_symbols = self._get_ofdm_subcarriers(modulated_symbols)
        logger.info("Parse OFDM symbols", num_symbols=psk_symbols.shape[0])

        # Extract useful subcarriers
        demodulated_bits = self.modulation.demodulate(psk_symbols.flatten())

        return np.packbits(demodulated_bits, bitorder='big')

if __name__ == "__main__":
    decoder = Decoder(modulation='bpsk')
    decoded_data = decoder.decode_file(Path("../text.txt.wav"))