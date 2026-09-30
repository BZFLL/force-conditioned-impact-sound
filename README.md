# Force Granularity in Neural Impact-Sound Generation

**Project page** (audio examples, full result tables): [bzfll.github.io/force-conditioned-impact-sound](https://bzfll.github.io/force-conditioned-impact-sound/)

Companion code for the paper *Time-Resolved Impact Force Improves Neural
Impact-Sound Generation*. Two generators are trained on four cumulative levels:
L1 (identity), L2 (peak), L3 (descriptor, six numbers) and L4 (frames,
time-resolved force). One is a filterbank DDSP (differentiable digital signal
processing) model, the other a mel-spectrogram CVAE (conditional variational
autoencoder). The data are ObjectFolder-Real (100 objects, 3,548 strikes,
seen- and unseen-object splits). The repository also holds the code of three
controls on L4. L4′ is L4 without the peak and the descriptor. The frame shuffle
gives each strike, at test time only, the 200 ms force window (starting 5 ms
before the force peak) of another test strike of the same split. The template pulse replaces every force window by the
average training pulse, scaled to the strike's own peak. For DDSP, L4 pools the
force window in blocks of 32 samples (0.67 ms). For frames at 2 ms and 6.7 ms,
which use blocks of 96 or 320 samples, the repository holds the result files only.

![FAD change relative to each generator's own L1 (identity) at L2, L3 and L4, for DDSP and CVAE generators on seen and unseen objects](docs/img/fig_ladder_main.png)

## Setup

```bash
pip install -r requirements.txt         # evaluation environment, Python 3.14
pip install -r requirements_train.txt   # training environment (GPU)
export PYTORCH_ENABLE_MPS_FALLBACK=1    # only on the MPS backend (macOS)
```

## Data

ObjectFolder-Real (Gao et al., CVPR 2023) is not redistributed; download it
from the [official project page](https://objectfolder.stanford.edu/) and
point the code at it:

```bash
export OFR_ROOT=/path/to/ObjectFolder_Real                # per-object impact folders
export OFR_OBJECTS_CSV=/path/to/ObjectFolder/objects.csv   # object -> material map
```

The splits used in the paper are in `splits/` (seed 42). `within` is the
seen-object split and `heldout` the unseen-object split.

## Training

```bash
python3 src/train.py --backbone ddsp --split within --ablation 4 --seed 42 \
        --save_dir experiments/ablation_4_within_ddsp_s42
```

`--ablation 1..4` selects the conditioning level, `--backbone {ddsp,cvae}`
the generator, `--split {within,heldout}` the seen/unseen-object split.
`scripts/run_grid_{ddsp,cvae}.sh` train the L1-L4 runs;
`scripts/run_grid_ladder_{within,heldout}.sh` and
`scripts/run_grid_pca6_within.sh` train the runs with feature sets `impulse`,
`v2A`, `v2AB` and `pca6` (table below). Each run directory under
`experiments/` carries the `config.json` it was trained with.

The two training variants of the paper's Table 2 (envelope loss and
pretraining; DDSP, level 4):

```bash
# envelope loss, both splits, seeds 42-44
python3 src/train.py --backbone ddsp --split heldout --ablation 4 --seed 42 \
        --env_lambda 8.8 --save_dir experiments/envloss/envloss_heldout_L4_s42
python3 src/train.py --backbone ddsp --split within --ablation 4 --seed 42 \
        --env_lambda 8.8 --save_dir experiments/envloss/envloss_within_L4_s42

# pretraining on synthetic noise bursts, then fine-tuning on the real splits
python3 src/make_synthetic_ir_pool.py   # one-time: 2,000 simulated room impulse responses
python3 src/train.py --backbone ddsp --split heldout --ablation 4 \
        --epochs 300 --synthetic \
        --save_dir experiments/pretrain/pretrain_synthetic
python3 src/train.py --backbone ddsp --split heldout --ablation 4 --seed 42 \
        --init_from experiments/pretrain/pretrain_synthetic/best_model.pt \
        --save_dir experiments/pretrain/ft_heldout_L4_s42
```

`scripts/run_grid_envloss_pretrain.sh` runs these commands (seeds 42-44, both
splits) except the second one, the seen-object envelope loss.

L4′ and the template pulse need their own training runs:

```bash
# L4′: identity and force frames, no peak and no descriptor (level 5 in the code);
# repeat for both splits and seeds 42-46
python3 src/train.py --backbone ddsp --split within --ablation 5 --seed 42 \
        --epochs 500 --batch_size 16 --n_bands 2048 --modal on \
        --save_dir experiments/ablation_5_within_ddsp_s42
bash scripts/run_grid_cvae_ablation5.sh    # CVAE, both splits, seeds 42-46

# template pulse (DDSP, level 4); the first line fits the template only if
# stats/template_pulse_<split>.pt is missing
python3 src/fit_template_pulse.py --split within
python3 src/train.py --backbone ddsp --split within --ablation 4 \
        --force_input template_peak --template_path stats/template_pulse_within.pt \
        --seed 42 --epochs 500 --batch_size 16 --n_bands 2048 --modal on \
        --save_dir experiments_template/template_4_within_ddsp_s42
bash scripts/run_grid_ddsp_template.sh     # both splits, seeds 42-46
```

The templates used in the paper are
`stats/template_pulse_{within,heldout}.pt`. A template-pulse run reads
`stats/force_curve_stats_<split>.pt`, which any DDSP training run on that split
writes (for example the L4 runs of `scripts/run_grid_ddsp.sh`). Template-pulse runs go to `experiments_template/`,
so that `src/evaluate.py` does not pool them with the L4 runs. The frame shuffle reuses the L4
checkpoints and needs no training.

Force-representation names used in the code and the run directories:

| Code name (`--feature_set`) | Paper name | Dim. | Run-dir suffix |
|---|---|---|---|
| `v1` | L3 (descriptor), computed on the full 6 s recording of the strike: peak, full width at half maximum, bounce count, max bounce ratio, decay constant, asymmetry; L2 (peak) uses its dimension 0 | 6 | none |
| `impulse` | impulse (integral of the force magnitude) | 1 | `_impulse` |
| `v2A` | windowed three-number descriptor (peak, contact duration, asymmetry) | 3 | `_v2A` |
| `v2AB` | windowed descriptor (`v2A` + impulse, bounce count, max bounce ratio; computed on the 200 ms force window) | 6 | `_v2AB` |
| `pca6` | PCA-6 descriptor (scores on a train-split PCA basis of the force curve) | 6 | `_pca6` |

Code names of L4′ and the template pulse:

| Code | Paper name | Run directories |
|---|---|---|
| `--ablation 5` | L4′ | `experiments/ablation_5_<split>_<backbone>_s<seed>` |
| `--ablation 4 --force_input template_peak` | template pulse | `experiments_template/template_4_<split>_ddsp_s<seed>` |

## Evaluation

Evaluation expects `best_model.pt` (written by `src/train.py`) in each run
directory under `experiments/`; run the training grids above first. The
run directories under `experiments/envloss/` and `experiments/pretrain/` record
their training configuration in `env_config.json`, `pretrain_config.json` or
`finetune_config.json`.

```bash
bash scripts/eval_test_main.sh               # Table 1, rows L1-L4: results/results_{ddsp,cvae}_<split>_test.json
python3 src/evaluate_ladder.py --split within --pca6   # Table 2 descriptor rows except L3 − L4: results/ladder_ddsp_within_test.json
python3 src/evaluate_ladder.py --split heldout         # results/ladder_ddsp_heldout_test.json
python3 src/plot_ladder_main.py              # Fig. 2, rows L2-L4: figures/fig_ladder_main.png
python3 src/eval_val_fad.py --split within   # pretraining, validation set
python3 src/eval_val_fad.py --split heldout  # envelope loss and pretraining, validation set
python3 src/modal_baseline_eval.py           # modal synthesis and retrieval, validation set
python3 src/whitenoise_baseline.py           # white noise, test set
```

`scripts/eval_test_main.sh` evaluates only the 80 L1-L4 runs; the L4′ and
Table 2 descriptor runs under `experiments/` do not enter Table 1.
`src/eval_val_fad.py` writes the validation-set values of pretraining (both
splits) and of the envelope loss (unseen objects) to `results/val_fad_<split>.json`.
`src/modal_baseline_eval.py` writes `results/modal_baseline_val.json`.
`src/whitenoise_baseline.py` writes `results/whitenoise_fad_test.json`.

The three controls (run each with `--split heldout` as well):

```bash
python3 src/eval_frames_only.py --split within                   # L4′, DDSP: results/frames_only_within_test.json
python3 src/eval_frames_only.py --split within --backbone cvae   # L4′, CVAE: results/frames_only_cvae_within_test.json
python3 src/force_shuffle_multi.py --split within --n-reassignments 10       # frame shuffle, DDSP: results/force_shuffle_multi_within_test.json
python3 src/force_shuffle_multi_cvae.py --split within --n-reassignments 10  # frame shuffle, CVAE: results/force_shuffle_multi_cvae_within_test.json
python3 src/eval_template_pulse.py --split within                # template pulse: results/template_pulse_within_test.json
```

`results/frame_rate_<split>_test.json` holds the results of frames at 2 ms and
6.7 ms. The code that trains and scores them is not in this repository.

If the VGGish encoder cannot be downloaded, the evaluation scripts stop, except
`src/evaluate.py`, which falls back to a log-mel encoder and marks outputs
`fad_encoder_compliant: false`. Such results are not comparable with the paper.

`pytest` runs the test suite (no dataset or checkpoints needed; one
template-pulse test skips without the dataset).
