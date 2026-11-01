# 5G LEO ISAC

Standalone research implementation for the **V18 submission**, with the final selected neural/statistical models, frozen physical receiver and eight evaluated configurations. The default source installation and version **0.2.0** delivery use V18. The older [r2 delivery](artifacts/r2) remains historical.

The adopted manuscript identities are recorded in [submission provenance](provenance/SNAPSHOT.json); manuscript files are not included or modified. No training, recalibration or campaign rerun was performed for this update.

## Install and verify

From the repository root (tested Linux/Python 3.10):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.lock
.venv/bin/python -m pip install --no-deps .
.venv/bin/python -m leo_ofdm_repro methods
.venv/bin/python -m leo_ofdm_repro verify-assets --assets ./selected_assets
.venv/bin/python -m leo_ofdm_repro reproduce
```

`reproduce` is CPU-only. It recomputes final/ever-correct and wrong-first outcomes on the frozen 512-flight/2048-attempt quality cohort, any-alert false alarms and exact one-sided bounds on the separate fixed 3000-record no-UAV cohort, and the recorded Nov 1 timing statistics. It includes the original paired whole-record bootstrap. It performs no inference, generation or new timing measurement. Wall-clock times are recorded results, not a promise of repeatable timing on another host/session.

For GPU execution, install `requirements-gpu.lock` instead of the CPU lock, adding `--extra-index-url https://download.pytorch.org/whl/cu124`. The pinned stack uses PyTorch 2.6.0+cu124 and Triton 3.2.0. External dependencies are not vendored. A prebuilt wheel and companion asset ZIP are in [artifacts/v18](artifacts/v18).

## Final model identity

| Selector | Original selected checkpoint SHA-256 | Shipped asset |
|---|---|---|
| N5 | `576b78e35a3d966199e588ca3bc9ee2209fe7cebc09023ea40208f93e9bd6be3` | `N5_step500_weights.pt`; model tensors only |
| S5 | `49b07366253aa2fcdc1f3c6ae08048e0d206ad5165dff7768bc2cba6183d626d` | `S5_linear_summary_alpha0.1.npz`; byte-identical |

The neural export removes optimizer/RNG metadata; its separate hash and exact tensor-identity proof are in [configuration provenance](provenance/V18_CONFIGURATION.json). The historical neural identifier `query_rich_step1000` maps to the selected step-500 weights; names are not used as proof of model identity. Normalization, rate prior, thresholds, precision, feature/preprocessing code, refinement, gate rules, scoring and selected literature adaptations match the evaluated source.

## Run the receiver

The selected methods are N5, S5, C9, E27, DS, TS1, TS2 and OLD KT+LVD. The existing N5/S5/C9 always27 component controls are also exposed explicitly.

```sh
.venv/bin/python -m leo_ofdm_repro paced --assets ./selected_assets --method N5 \
  --reference ./raw/reference.npy --observation ./raw/observation.npy \
  --alignment ./raw/alignment.npy --output ./checks/N5-paced.json
```

Inputs are one or four seconds of complex64 raw samples at 30.72 MHz, a preceding 10 ms reference guard and supplied 20 ms alignment rows. The original scheduler performs cancellation/correlation/preparation, physical searches, selection/refinement and emitted outputs. Early alerts may be updated or withdrawn at the final check. An earlier alert forces the full 27-rate final search; otherwise the frozen 15-rate rejection gate may stop with no alert.

The timing boundary starts with host-resident digital samples after initialization/warmup; disk/RF/ADC/acquisition/alignment estimation are excluded. To follow the submission warmup protocol, pass all three `--warmup-reference`, `--warmup-observation`, `--warmup-alignment` arrays from a disjoint calibration recording. Without them, the command explicitly reports warmup on its receiver input. `prepared` is a separate one-second correlation/variance interface with `timing_valid=false`.

[Running instructions](docs/RUNNING.md) cover these commands, frozen input export, waveform/geometry/raw generation, units and scoring. Compact scene/seed manifests are included; bulky generated waveforms, raw recordings and caches are excluded. MATLAB/5G Toolbox are external requirements for NR generation.

## Scope of evidence

Quality uses exactly the sealed original 256 plus new 256 independent four-second UAV recordings (2048 dependent attempts), excluding unsealed extras. FA uses 3000 fresh independent one-second no-UAV recordings; any emitted alert counts once, including withdrawals. The earlier 300 short and 16 long empties are not pooled into that endpoint. Timing uses the final-receiver Nov 1 panel of 16 UAV plus eight no-UAV four-second records, with separate existing component controls and whole-record paired uncertainty.

The bounded update checks compare final-model logits, physical predictions, scores, candidate choices, gate branches and alerts against the frozen implementation, with source/install/artifact checks recorded in [validation](validation/CHECKPOINT.md). Saved-outcome recomputation is distinct from rerunning the physical campaign. Full 402-frame waveform recovery and full-cohort raw regeneration were not repeated; new smoke checks do not establish full-cohort detection rates or matched hardware timing.

The original package/history, campaigns, results and manuscript remain unchanged. This repository preserves its initial r2 commit and adds a normal submission-update commit. [Snapshot verification](tools/verify_snapshot.py), [provenance](provenance/V18_CONFIGURATION.json) and [scientific attribution/adaptations](THIRD_PARTY_NOTICES.md) are included. No new open-source license was selected; see [rights](RIGHTS.md).
