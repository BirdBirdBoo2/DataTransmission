import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import structlog
from scipy.io import wavfile as wav
from .psk_modulation import QPSK_Modulation, Modulation, BPSK_Modulation, PSK_Modulation

DO_PLOTS = False

logger = structlog.stdlib.get_logger(__name__)


class Encoder:
    def __init__(self,
                 n_subcarriers: int = 64,
                 cyclic_prefix_length: int = 16,
                 modulation: str = 'qpsk',
                 M: int = 4,
                 grace_period_s: float = 0.0,
                 ofdm_frequency_hz: float = 512,
                 carrier_frequency_hz: float = 2048,
                 DEBUG_use_carrier: bool = True):
        self.ofdm_frequency_hz = ofdm_frequency_hz
        self.carrier_frequency_hz = carrier_frequency_hz
        self.sample_rate = int(4 * carrier_frequency_hz)

        self.n_subcarriers = n_subcarriers  # Number of subcarriers
        self.cyclic_prefix_length = cyclic_prefix_length  # Cyclic prefix length
        self.grace_period_s = grace_period_s
        self.DEBUG_use_carrier = DEBUG_use_carrier

        self.modulation: Modulation
        if modulation == 'qpsk':
            self.modulation = QPSK_Modulation(n_subcarriers)
        elif modulation == 'bpsk':
            self.modulation = BPSK_Modulation(n_subcarriers)
        elif modulation == 'mpsk':
            self.modulation = PSK_Modulation(M)
        else:
            raise ValueError(f"Invalid modulation type: {modulation}")

    def read_bytes(self, data: bytearray):
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    def _make_symbol(self, ofdm_frame: np.ndarray) -> np.ndarray:
        assert self.ofdm_frequency_hz % self.n_subcarriers == 0, "OFDM frequency must be a multiple of the number of subcarriers"

        base_frequency = self.ofdm_frequency_hz // self.n_subcarriers

        min_symbol_duration_s = 1 / base_frequency

        symbol = np.zeros(self.sample_rate, dtype=np.complex128)
        for i in range(self.n_subcarriers):
            symbol[base_frequency * (i + 1)] = ofdm_frame[i]
        symbol[-self.sample_rate // 2:] = np.conj(np.flip(symbol[1:self.sample_rate // 2 + 1]))

        time_domain_symbol = np.fft.ifft(symbol)
        actual_ofdm_symbol_count = int(min_symbol_duration_s * self.sample_rate)
        time_domain_symbol = time_domain_symbol[:actual_ofdm_symbol_count]

        return time_domain_symbol

    def _make_symbols(self, ofdm_frames: np.ndarray) -> np.ndarray:
        base_frequency = self.ofdm_frequency_hz // self.n_subcarriers
        min_symbol_duration_s = 1 / base_frequency
        logger.info("Min symbol duration", min_symbol_duration_s=min_symbol_duration_s)
        actual_ofdm_symbol_count = int(min_symbol_duration_s * self.sample_rate)

        symbols = np.zeros((ofdm_frames.shape[0], actual_ofdm_symbol_count), dtype=np.complex128)

        for i in range(ofdm_frames.shape[0]):
            symbols[i] = self._make_symbol(ofdm_frames[i])

        return symbols

    def ofdm_modulate(self, symbols: np.ndarray[np.complex128]):
        num_symbols = symbols.shape[0]
        num_sub = self.n_subcarriers
        cyclic_prefix_length = self.cyclic_prefix_length

        assert num_symbols % num_sub == 0, "Number of symbols must be a multiple of the number of subcarriers"

        ofdm_frames = symbols.reshape(-1, num_sub)

        time_domain_symbols = self._make_symbols(ofdm_frames)

        if DO_PLOTS:
            plt.figure(figsize=(15, 5))
            for i in range(time_domain_symbols.shape[0]):
                plt.plot(np.real(time_domain_symbols[i]), label=f'real_{i}')
                break
                # plt.plot(np.imag(time_domain_symbols[i]), label=f'imag_{i}')
            plt.title("Time domain symbols")
            plt.legend()
            plt.show()

        time_domain_symbols = np.real(time_domain_symbols)

        ofdm_frame_len = time_domain_symbols.shape[1]
        # cp_in_sr_length = int(cyclic_prefix_length / num_sub * ofdm_frame_len)
        cp_in_sr_length = 0

        if cp_in_sr_length > 0:
            # Add cyclic prefix
            cp = time_domain_symbols[:, -cp_in_sr_length:]
            time_domain_symbols = np.hstack([cp, time_domain_symbols])

        if self.grace_period_s > 0:
            len_grace = int(self.grace_period_s * self.sample_rate)
            grace = np.ones((time_domain_symbols.shape[0], len_grace), dtype=float) * 0.002
            time_domain_symbols = np.hstack([time_domain_symbols, grace])

        time_domain_signal = time_domain_symbols.flatten()

        carrier_signal = self._make_carrier_waveform(time_domain_signal)
        logger.info(f"Carrier signal created", carrier_frequency=self.carrier_frequency_hz)
        logger.info("OFDM frames computed", ofdm_frame_len=ofdm_frame_len)

        if DO_PLOTS:
            plt.figure(figsize=(25, 5))
            plt.plot(np.real(time_domain_signal), label='real')
            plt.axvline(x=cp_in_sr_length, color='r', linestyle='--', label='cp')
            plt.plot(carrier_signal, linestyle='--', label='carrier')

            plt.plot(time_domain_signal * carrier_signal, label='carrier')

            plt.legend()
            plt.tight_layout()
            plt.show()

        return time_domain_signal * carrier_signal if self.DEBUG_use_carrier else time_domain_signal

    def _make_carrier_waveform(self, time_domain_signal):
        carrier_signal_t = np.arange(0, len(time_domain_signal))
        carrier_signal = np.sin(2 * np.pi * carrier_signal_t * self.carrier_frequency_hz / self.sample_rate)
        return carrier_signal

    def save_waveform(self, signal, output_file: Path):
        signal = np.real(signal)
        signal = np.int16(signal / np.max(np.abs(signal)) * 32767)  # Normalize to int16 (required by the WAV format)
        wav.write(output_file, self.sample_rate, signal)

    def encode(self, data: bytearray):
        logger.info("Read data to encode", data_len_bytes=len(data))
        bits = self.read_bytes(data)
        logger.info("Converted data to bits", data_len_bits=len(bits))
        symbols = self.modulation.modulate(bits)
        logger.info("Modulated bits to symbols", data_len_symbols=len(symbols))
        signal = self.ofdm_modulate(symbols)
        logger.info("Modulated symbols to signal", signal_pts=len(signal))
        return signal

    def encode_to_file(self, data: bytearray, output_file: Path):
        signal = self.encode(data)
        self.save_waveform(signal, output_file=output_file)
        print(f"Encoded data into {output_file}")
