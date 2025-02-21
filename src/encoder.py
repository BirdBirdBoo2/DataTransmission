import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import structlog
from scipy.io import wavfile as wav
from .psk_modulation import QPSK_Modulation, Modulation, BPSK_Modulation, PSK_Modulation

DO_PLOTS = False
DEFAULT_SAMPLE_RATE = 44100

logger = structlog.stdlib.get_logger(__name__)


class Encoder:
    def __init__(self, n_subcarriers: int = 64, cyclic_prefix_length: int = 16, modulation: str = 'qpsk', M: int = 4):
        self.n_subcarriers = n_subcarriers  # Number of subcarriers
        self.cyclic_prefix_length = cyclic_prefix_length  # Cyclic prefix length

        self.modulation: Modulation
        if modulation == 'qpsk':
            self.modulation = QPSK_Modulation()
        elif modulation == 'bpsk':
            self.modulation = BPSK_Modulation(n_subcarriers)
        elif modulation == 'mpsk':
            self.modulation = PSK_Modulation(M)
        else:
            raise ValueError(f"Invalid modulation type: {modulation}")

    def read_bytes(self, data: bytearray):
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    def ofdm_modulate(self, symbols: np.ndarray[np.complex128]):
        num_symbols = symbols.shape[0]
        num_sub = self.n_subcarriers
        cyclic_prefix_length = self.cyclic_prefix_length

        assert num_symbols % num_sub == 0, "Number of symbols must be a multiple of the number of subcarriers"

        ofdm_frames = symbols.reshape(-1, num_sub)
        num_ofdm_frames = ofdm_frames.shape[0]

        # Create OFDM symbol (using Hermitian symmetry for real signal)
        superresolution = 4
        ofdm_symbols = np.zeros((num_ofdm_frames, 2 * superresolution * num_sub + 1), dtype=np.complex128)
        ofdm_symbols[:, 1:num_sub + 1] = ofdm_frames
        ofdm_symbols[:, -num_sub:] = np.conj(np.flip(ofdm_frames, axis=1))

        time_domain_symbols = np.fft.ifft(ofdm_symbols, axis=1)

        if DO_PLOTS:
            plt.figure(figsize=(15, 5))
            for i in range(time_domain_symbols.shape[0]):
                plt.plot(np.real(time_domain_symbols[i]), label=f'real_{i}')
                plt.plot(np.imag(time_domain_symbols[i]), label=f'imag_{i}')
            plt.title("Time domain symbols")
            plt.legend()
            plt.show()

            ofdm_recovered = np.fft.fft(time_domain_symbols, axis=1)
            np.testing.assert_almost_equal(ofdm_recovered, ofdm_symbols, decimal=3)

        # Add cyclic prefix
        cp = time_domain_symbols[:, -cyclic_prefix_length * superresolution:]
        # cp = np.zeros_like(cp)
        time_domain_signal = np.hstack([cp, time_domain_symbols]).flatten()

        if DO_PLOTS:
            plt.figure(figsize=(25, 5))
            plt.axvline(cyclic_prefix_length * superresolution, color='r', linestyle='--')
            plt.axvline((2 * num_sub + 1 + cyclic_prefix_length) * superresolution, color='r', linestyle='--')
            plt.plot(np.real(time_domain_signal), label='real')
            plt.tight_layout()
            plt.show()

        carrier_signal_t = np.arange(0, time_domain_signal.shape[0])
        carrier_signal = np.sin(2 * np.pi * carrier_signal_t / 32) * 4

        return time_domain_signal * carrier_signal

    def save_waveform(self, signal, output_file: Path, sample_rate=DEFAULT_SAMPLE_RATE):
        signal = np.real(signal)
        signal = np.int16(signal / np.max(np.abs(signal)) * 32767)  # Normalize to int16 (required by the WAV format)
        wav.write(output_file, sample_rate, signal)

    def encode(self, data: bytearray):
        logger.info("Read data to encode", data_len_bytes=len(data))
        bits = self.read_bytes(data)
        logger.info("Converted data to bits", data_len_bits=len(bits))
        symbols = self.modulation.modulate(bits)
        logger.info("Modulated bits to symbols", data_len_symbols=len(symbols))
        signal = self.ofdm_modulate(symbols)
        logger.info("Modulated symbols to signal", signal_pts=len(signal))
        return signal

    def encode_to_file(self, data: bytearray, output_file: Path, sample_rate=DEFAULT_SAMPLE_RATE):
        signal = self.encode(data)
        self.save_waveform(signal, sample_rate=sample_rate, output_file=output_file)
        print(f"Encoded data into {output_file}")
