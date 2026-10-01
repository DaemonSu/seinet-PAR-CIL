# PAR-CIL

Reference implementation of **PAR-CIL**, a class-incremental specific emitter
identification method based on fixed-coordinate multi-center prototype memory
and synchronous shared projection.

The incremental stages use only the current-stage labeled samples. Historical
raw IQ samples and per-sample feature caches are not replayed. For each seen
class, the retained state contains one normalized route mean and `M` K-means
support centers in the frozen E1 feature space. The current E2 mapping is
applied synchronously to queries and all retained centers before cosine
classification and prototype supervision.

## Protocols

The paper configuration uses 100 devices with `50 + 5 x 10` enrollment stages.
Each class contributes 200 training, 50 validation, and 50 test examples.
Training at an incremental stage reads only that stage's five new classes;
validation and test sets are cumulative over all classes seen so far.

- ADS-B input: precomputed `adsb2d` tensors with shape `(29, 42, 3)`.
- WiSig input: four consecutive 256-IQ packets, yielding 1024-IQ examples and
  `q`, `abs(q)`, and `abs(FFT(q))` channels.
- WiSig raw capture regions are split before window construction. Training may
  overlap within its own region; validation/test windows are non-overlapping,
  and no raw packet is shared across splits.

See [docs/DATA_PROTOCOL.md](docs/DATA_PROTOCOL.md) for the directory contract.
The original datasets and trained checkpoints are not redistributed.

## Environment

```bash
conda create -n parcil python=3.10 -y
conda activate parcil
pip install -r requirements.txt
```

The reported experiments used CUDA. CPU execution is suitable for preprocessing
and unit tests but is not practical for full training.

## Data preparation

ADS-B tensors:

```bash
python code/28_prepare_adsb2d_cil.py \
  --source_root /path/to/adsb2d_samples \
  --output_root data/adsb_50_5x10
```

WiSig-1024 requires a previously fixed 100-device label mapping in
`REFERENCE/protocol.json`:

```bash
python code/142_prepare_wisig1024_block_split.py \
  --raw_source /path/to/wisig \
  --reference /path/to/reference_protocol \
  --code code \
  --output data/wisig1024_50_5x10
```

## Training

Step0 checkpoints must contain `encoder.pth`, `classifier.pth`, and `args.json`.
For ADS-B, train Step0 with:

```bash
python code/train_initial_adsb.py \
  --backbone adsb2d \
  --train_dir data/adsb_50_5x10/train-closed \
  --val_dir data/adsb_50_5x10/val-closed \
  --save_dir runs/adsb_seed42_step0 \
  --num_classes 50 --seed 42
```

Run the paper configuration with the validation folders supplied as the
stage-selection inputs. The final evaluation is performed separately on the
cumulative test folders:

```bash
python code/20_train_anchor_tree_mlp_noreplay.py \
  --init_dir runs/adsb_seed42_step0 \
  --save_dir runs/adsb_seed42_parcil_m32 \
  --data_root data/adsb_50_5x10 \
  --test_closed val-closed --test_add_template 'val-add{step}' \
  --initial_classes 50 --increment_size 5 --num_increments 10 \
  --centers_per_class 32 --e2_mode trainable --projector_alpha 0.95 \
  --new_ce_weight 1 --prototype_ce_weight 4 \
  --distill_weight 0 --new_data_distill_weight 0 \
  --anchor_ce_weight 0 --anchor_margin_weight 0 --seed 42
```

`100_train_fixed_memory_control.py` and `134_locked_test_evaluation.py` are the
audited control and locked-test drivers used for the manuscript. They verify
the data-access ledger, frozen E1 state, immutable old centers, checkpoint
hashes, and validation/test separation.

The historical filename `20_train_anchor_tree_mlp_noreplay.py` is retained so
that archived experiment configurations remain loadable; the implemented
method is PAR-CIL and does not use a tree classifier.

## Verification

```bash
python -m compileall -q code tests
pytest -q
```

No license has been selected for this repository yet. Dataset licenses remain
with their respective providers.
