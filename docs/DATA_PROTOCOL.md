# Data protocol

Prepared samples are named `device_<label>_<index>.npy` and organized as:

```text
DATA_ROOT/
  train-closed/   # classes 0-49, 200 samples/class
  val-closed/     # classes 0-49, 50 samples/class
  test-closed/    # classes 0-49, 50 samples/class
  train-add1/     # classes 50-54 only
  val-add1/       # classes 0-54 (cumulative)
  test-add1/      # classes 0-54 (cumulative)
  ...
  train-add10/    # classes 95-99 only
  val-add10/      # classes 0-99 (cumulative)
  test-add10/     # classes 0-99 (cumulative)
  protocol.json
```

Incremental optimization must not read `train-closed` or an earlier
`train-add*` directory. Old classes are represented only by the persistent
class-level bank built in frozen E1 coordinates.

## ADS-B

`28_prepare_adsb2d_cil.py` expects float32 tensors of shape `(29, 42, 3)`.
It selects disjoint train/validation/test files per device and records the raw
to incremental label mapping in `protocol.json`.

## WiSig-1024

`142_prepare_wisig1024_block_split.py` expects the official per-day pickle
layout. For every device/date capture it reserves the first 40% of packets for
training, the next 30% for validation, and the final 30% for testing before
constructing windows. Four packets form each 1024-IQ input. Candidate strides
are one packet for training and four packets for validation/test.

The script emits source hashes, tensor hashes, packet coverage, and an overlap
audit. A nonzero cross-split overlap causes preparation to fail.

