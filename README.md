# 5G LEO ISAC

Private source snapshot of the latest existing packaged simulator, receiver, selected models and eight baseline/configuration routes. This is package **r2**, committed on 2026-10-28T17:09:03+00:00, source package commit `0bd62cd6dfb2a53dbb2d30ddab711282cf1511f5`. No scientific code was rebuilt and no experiments were rerun for this repository upload.

**Model-version warning:** this preserves the packaged r2 assets from 28 October 2026. They predate the final paper's later selector training. Both N5 and S5 checkpoint hashes differ from the final selected models. This repository does **not** claim exact reproduction of the final V18 paper configuration. Updating the package/models is a separate future change; no final checkpoint has been silently substituted. Exact identities are in [snapshot provenance](provenance/SNAPSHOT.json).

| Model | Packaged checkpoint SHA-256 | Final selected checkpoint SHA-256 |
|---|---|---|
| N5 | `1a2432ed38cb4dfd7ec3b92b54baa50cc038f2cad0965acb5b43169c5c098b03` | `576b78e35a3d966199e588ca3bc9ee2209fe7cebc09023ea40208f93e9bd6be3` |
| S5 | `947b946d746d5be1502c9f5d36119b0ce3e698bb2f295cfc4a2f429bbcb4a574` | `49b07366253aa2fcdc1f3c6ae08048e0d206ad5165dff7768bc2cba6183d626d` |

The final N5 checkpoint is `N5_step500.pt`; the final model specification maps the historical `query_rich_step1000` identifier to that file. The historical identifier alone is not a checkpoint identity.

## Install

From the repository root, using Python 3.10:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.lock
.venv/bin/python -m pip install --no-deps .
.venv/bin/python -m leo_ofdm_repro methods
.venv/bin/python -m leo_ofdm_repro example
.venv/bin/python -m leo_ofdm_repro verify-assets --assets ./selected_assets
```

For the GPU receiver, install `requirements-gpu.lock` instead of the CPU lock, adding `--extra-index-url https://download.pytorch.org/whl/cu124`. The pinned stack uses PyTorch 2.6.0+cu124 and Triton 3.2.0. Dependency/build downloads use their normal package sources; dependencies are not vendored.

The unmodified prebuilt r2 wheel, asset ZIP, original instructions and exact hash manifest are in [artifacts/r2](artifacts/r2). The source and extracted seven-file asset bundle match that delivery. Run `python3 tools/verify_snapshot.py` for static snapshot/artifact verification; it performs no inference or generation.

## Receiver and simulation

The selected routes are N5, S5, C9, E27, DS, TS1, TS2 and the selected OLD KT+LVD adaptation. No baseline, frozen threshold, scoring rule, model or numerical operation is changed in this snapshot. [Running instructions](docs/RUNNING.md) cover recorded input export, geometry, raw generation, MATLAB waveform requirements, prepared execution, sample-paced execution and offline scoring.

```sh
.venv/bin/python -m leo_ofdm_repro paced --assets ./selected_assets --method N5 \
  --reference ./raw/reference.npy --observation ./raw/observation.npy \
  --alignment ./raw/alignment.npy --output ./checks/N5-paced.json
```

`paced` accepts one- or four-second raw complex64 recordings, the 10 ms reference guard and supplied 20 ms alignment rows. Its timing boundary is host-resident digital samples after initialization/warmup, excluding disk/RF/ADC/acquisition/alignment estimation. `prepared` consumes one second of correlations/variances and reports `timing_valid=false`. Neither interface silently generates datasets.

## Recorded validation and limits

The r2 [checkpoint](validation/CHECKPOINT.md) and receipts record 20 standalone tests, 40 saved scoring examples, geometry/native checks, short NR/raw generation, current-source agreement for all eight paced routes on one synthetic input, and installed execution in fresh CPU/GPU dependency environments with source reads denied. These are the existing October 2026 checks, not new experiments performed for this upload.

Full 402-frame waveform/hash recovery, full recording generation, four-attempt concurrency execution and matched paper-performance reproduction remain unexecuted in that bounded package validation. The existing sliced generation/pacing checks do not establish detection probabilities or matched final-paper latency.

The repository contains only the reviewed standalone snapshot, selected frozen assets, tests, notices, instructions, provenance and the existing small delivery artifacts. It excludes the original development Git history, authoring/vendor bridge, credentials, virtual environments, caches, generated raw/waveform/result datasets, private discussion and manuscript source. Original package history, source repositories, results and delivery artifacts remain untouched locally.

This repository is private. No public release or open-source license choice is implied. See [rights](RIGHTS.md) and [scientific attribution/adaptations](THIRD_PARTY_NOTICES.md).
