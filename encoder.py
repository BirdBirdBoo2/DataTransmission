import numpy as np
from pathlib import Path
from scipy.io import wavfile as wav

DEFAULT_SAMPLE_RATE = 44100


class Encoder:
    def __init__(self, n_subcarriers: int = 64, cyclic_prefix_length: int = 16):
        self.n_subcarriers = n_subcarriers  # Number of subcarriers
        self.cyclic_prefix_length = cyclic_prefix_length  # Cyclic prefix length
        self.qpsk_mapping = {
            (0, 0): 1 + 1j,
            (0, 1): -1 + 1j,
            (1, 0): 1 - 1j,
            (1, 1): -1 - 1j,
        }

    def read_file(self, file_path):
        with open(file_path, 'rb') as f:
            data = f.read()
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    def qpsk_modulate(self, bits):
        bit_pairs = bits.reshape(-1, 2)
        symbols = np.array([self.qpsk_mapping[tuple(b)] for b in bit_pairs])
        return symbols

    def ofdm_modulate(self, symbols):
        num_symbols = len(symbols)
        num_sub = self.n_subcarriers
        cyclic_prefix_length = self.cyclic_prefix_length

        num_ofdm_symbols = int(np.ceil(num_symbols / (num_sub // 2 - 1)))

        # Zero-pad symbols to fit into OFDM frames
        padded_symbols = np.concatenate(
            [symbols, np.zeros(num_ofdm_symbols * (num_sub // 2 - 1) - num_symbols, dtype=complex)])

        # Reshape into OFDM frames
        ofdm_frames = padded_symbols.reshape(num_ofdm_symbols, -1)

        # Create OFDM symbol (using Hermitian symmetry for real signal)
        ofdm_symbols = np.zeros((num_ofdm_symbols, num_sub), dtype=complex)
        ofdm_symbols[:, 1:num_sub // 2] = ofdm_frames
        ofdm_symbols[:, -num_sub // 2 + 1:] = np.conj(np.flip(ofdm_frames, axis=1))

        time_domain_symbols = np.fft.ifft(ofdm_symbols, axis=1)

        # Add cyclic prefix
        cp = time_domain_symbols[:, -cyclic_prefix_length:]
        time_domain_signal = np.hstack([cp, time_domain_symbols]).flatten()
        return time_domain_signal

    def save_waveform(self, signal, output_file: Path, sample_rate=DEFAULT_SAMPLE_RATE,):
        signal = np.real(signal)
        signal = np.int16(signal / np.max(np.abs(signal)) * 32767)  # Normalize to int16 (required by the WAV format)
        wav.write(output_file, sample_rate, signal)

    def encode(self, file_path, output_file: Path, sample_rate=DEFAULT_SAMPLE_RATE):
        bits = self.read_file(file_path)
        symbols = self.qpsk_modulate(bits)
        signal = self.ofdm_modulate(symbols)
        self.save_waveform(signal, sample_rate=sample_rate, output_file=output_file)
        print(f"Encoded {file_path} into {output_file}")
