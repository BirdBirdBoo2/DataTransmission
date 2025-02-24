from dataclasses import dataclass
from typing import Iterable, TypeVar, Generic, cast
import numpy as np

T = TypeVar('T')


@dataclass
class Range(Generic[T]):
    start: int
    end: int
    value: T

    @property
    def length(self):
        return self.end - self.start


def _extract_same_value_ranges(arr: np.ndarray) -> Iterable[Range[T]]:
    if len(arr) == 0:
        return

    curr_range = Range(0, 0, arr[0])
    for i in range(1, len(arr)):
        curr_range.end = i - 1

        if arr[i] != arr[i - 1]:
            yield curr_range
            curr_range = Range(i, i, arr[i])

    yield curr_range


def extract_loud_signal_ranges(signal: np.ndarray, threshold: float, averaging_window_len: int) \
        -> tuple[Iterable[Range[bool]], np.ndarray]:
    loudness = np.abs(signal)
    averaging_mask = np.ones(averaging_window_len)
    averaging_mask /= averaging_mask.shape[0]
    loudness: np.ndarray = np.convolve(loudness, averaging_mask, mode='same')
    signal_detect: np.ndarray[bool] = loudness > threshold

    return list(_extract_same_value_ranges(signal_detect)), loudness
