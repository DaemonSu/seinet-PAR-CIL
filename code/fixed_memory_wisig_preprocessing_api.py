"""WiSig packet normalization and q/abs(q)/FFT feature construction."""

import math

import numpy as np


def as_iq_channels(array: np.ndarray) -> np.ndarray:
    x = np.asarray(array)
    if np.iscomplexobj(x):
        x = x.reshape(-1)
        return np.stack([x.real, x.imag], axis=0).astype(np.float32, copy=False)
    if x.ndim == 1:
        if x.size % 2:
            raise ValueError(f"Cannot interpret odd-length IQ vector: {x.shape}")
        return x.reshape(-1, 2).T.astype(np.float32, copy=False)
    if x.ndim == 2 and x.shape[0] == 2:
        return x.astype(np.float32, copy=False)
    if x.ndim == 2 and x.shape[1] == 2:
        return x.T.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported IQ shape: {x.shape}")


def normalize_wisig_packets(packets: np.ndarray) -> np.ndarray:
    packets = np.asarray(packets, dtype=np.float32)
    if packets.ndim != 3 or packets.shape[-1] != 2:
        raise ValueError(f"Expected (packets, IQ, 2), got {packets.shape}")
    power = np.sqrt(np.mean(np.sum(packets.astype(np.float64) ** 2, axis=-1), axis=-1))
    power = np.maximum(power, 1e-12)
    return (packets / power[:, None, None]).astype(np.float32, copy=False)


def to_q_abs_fft(iq: np.ndarray) -> np.ndarray:
    q = as_iq_channels(iq)[1]
    return np.stack([q, np.abs(q), np.abs(np.fft.fft(q))], axis=0).astype(
        np.float32, copy=False
    )


def make_wisig_segment(arr: np.ndarray, start_packet: int, segment_len: int) -> np.ndarray:
    packet_len = int(arr.shape[1])
    packets_needed = int(math.ceil(segment_len / packet_len))
    flat = arr[start_packet : start_packet + packets_needed].reshape(-1, 2)
    flat = flat[:segment_len]
    if len(flat) < segment_len:
        padded = np.zeros((segment_len, 2), dtype=np.float32)
        padded[: len(flat)] = flat
        flat = padded
    return to_q_abs_fft(flat)

