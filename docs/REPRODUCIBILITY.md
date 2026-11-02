# Results, fixed inputs and reproducibility

The repository supports three distinct operations: recomputing statistics from stored outputs, running receiver inference on supplied arrays, and generating new arrays from fixed scene and waveform seeds. The CPU quick start exercises the first operation. Receiver execution and complete signal regeneration require the additional inputs and tools in [usage](USAGE.md).

## Included evaluation records

| Evaluation | Independent records | Observations per record | Stored content |
|---|---|---|---|
| Detection quality | 512 simulated four-second UAV recordings | Four nonoverlapping one-second attempts | Emitted alerts, endpoint errors and final/ever-correct outcomes for all eight main receivers. |
| False alerts | 3000 separate one-second no-UAV recordings | One attempt | Any-alert and final-alert indicators for all eight receivers. |
| Timing | 16 UAV and eight no-UAV four-second recordings | Four dependent attempts | Emitted predictions and measured availability/terminal times, plus three full-final-search component controls on four records. |

The four exported quality parts contain 16, 52, 188 and 256 independent recordings. Together they give the 512-record quality population. Older false-alert recordings and unused extension samples are excluded from these endpoints. Timing is a separate contemporaneous panel, rather than a timing summary pooled from quality runs.

A **scene** fixes the satellite/UAV trajectory, steering and waveform identity. A manifest **episode** is one recording, adding its radar cross section and noise seeds to that scene. A **split** is the manifest's grouping identifier. The full manifest contains every scene, episode and waveform entry. Exported `scene.json` and `waveform.json` are convenience copies of the first listed entries, not a replacement for the full cohort. Scene and recording identifiers label the preserved relationships and do not determine the stored numeric seeds.

All receivers use a fixed library of nineteen NR waveforms with different cell identities and payload/control seeds. The reported variability is conditional on this library. Geometry, waveform identity, scene/noise seeds and recording grouping are retained in the input manifests. The scene assumes line-of-sight propagation, modeled leakage, acquired beams and supplied reference alignment. It does not establish performance under buildings, ground reflections or acquisition uncertainty.

Export metadata with:

```sh
.venv/bin/python -m fiveg_leo_isac cohorts
.venv/bin/python -m fiveg_leo_isac export-inputs \
  --cohort quality-part1 --waveform-root "$PWD/work/waveforms" --output work/quality-inputs
```

`quality-part1` through `quality-part4` cover the quality population. `no-uav` exports the 3000-record false-alert population. `calibration` exports the 60 four-second no-UAV calibration records. `example` exports one existing UAV scene. These commands write metadata only. The exported no-UAV test manifest is opened for explicit post-evaluation regeneration while preserving its scenes and seeds. It does not authorize a new adaptive test selection.

## Endpoint definitions

An alert is correct when its delay error is at most `30.72e6 / 18.36e6` samples (about 54.5 ns) and its Doppler error is at most 1 Hz, both evaluated at the end of the samples used by that check. The one-second nominal resolution limits also apply to early checks.

- **Final correct detections** count UAV attempts whose final report is a correct alert.
- **Ever correct** counts attempts with at least one correct emitted alert, even if the final output is wrong or withdrawn.
- **Wrong first alert** counts attempts whose first positive alert is incorrect. Later correction does not remove that event.
- **Any false alert** counts a no-UAV recording once if any alert is emitted, including one later withdrawn. The reported upper probability bound is exact one-sided 95% Clopper–Pearson.

`results` returns these measures as counts, probabilities or fields explicitly ending in `_pct`. The JSON does not convert probabilities into percentages implicitly.

## Timing and uncertainty

The paper's timing panel was run on an NVIDIA A10-24Q with 24 GB memory, Python 3.10.12, PyTorch 2.6.0/CUDA 12.4 and Triton 3.2.0. CPU threads are fixed at four and TF32 is disabled. Receivers were run separately after model loading and warmup with a disjoint calibration input.

Time starts when the first host-resident digital sample is released. It includes sample arrival, CPU-to-GPU transfer, leakage suppression, correlation, preparation, model inference, physical searches, refinement and queued work. Disk reads, model/JIT loading, warmup, RF reception, ADC and beam/reference acquisition are excluded.

**First-correct UAV alert time** is averaged only over UAV attempts with at least one correct report. A failed attempt does not contribute to that conditional mean. **Combined mean time** instead gives each UAV attempt its first-correct time or, if never correct, its terminal decision time. No-UAV attempts contribute terminal time. UAV and no-UAV class means then receive equal weight, independently of their sample counts.

Confidence intervals resample entire four-second recordings, keeping four dependent attempts together. The same sampled record indices are paired across receivers. UAV and no-UAV records are resampled separately. The supplied calculation uses 20,000 bootstrap replicates with seed 20261101 and reports 2.5th/97.5th percentiles. These intervals describe record variability within the measured session, not variation between hardware sessions.

```sh
.venv/bin/python -m fiveg_leo_isac results
```

This command recalculates the stored counts, means and intervals and checks them against the included reference aggregates. It does not run new inference or timing. Raw recorded outputs remain in compressed package data, including the full-final-search control records. The standard summary reports the eight main receivers.

## Models and fixed settings

| File under `selected_assets/` | SHA-256 |
|---|---|
| `models/neural_selector.pt` | `b92a337ab28a6690caaef1a361a882a063e8d8fffef54918849d7e3c632748ca` |
| `models/statistical_selector.npz` | `49b07366253aa2fcdc1f3c6ae08048e0d206ad5165dff7768bc2cba6183d626d` |
| `models/normalization.npz` | `989c6605984d833f76121f629ce5c602195d1450515931bd3798b45d5b15030a` |
| `policies/phase_refinement/rate_prior.npy` | `2cef7cabfe2919b6f626be72d18ab565204bec13d857fee3a28579b61a151a49` |

The neural file contains 42 model tensors without optimizer or random-state metadata. The statistical file contains the fitted affine predictor. Model fitting used 35,712 examples from 216 independent UAV flights, with selection on 4,608 examples from 32 separate development flights. Training used UAV-present examples to learn search selection. Training scripts and training data are not included.

`manifest.json` binds the seven asset files and their sizes/hashes. `physical_configuration.json` records geometry and physical settings. The receiver configuration records alert thresholds, early search budgets, gate settings and two-step policies. The refined-score alert threshold is `28.148725509643555` for the selective, exhaustive and dual-scale receivers. The physical rejection gate also retains its separate cutoff `28.41327667236328` and its conjunction with the alert-threshold comparison. The two-step early-look lower/upper thresholds are rounded upward to representable float32 values before execution. No command recalibrates these settings.

The component controls `adaptive-neural-full-final`, `adaptive-statistical-full-final` and `adaptive-classical-full-final` preserve their corresponding early searches and force all 27 rates at the final check. They isolate the final rejection shortcut.

## Validation scope

Twenty-three tests cover alert scoring, asset integrity, raw-input forms and timestamp causality. All eight main receivers and three full-final controls have numerical comparisons on two saved synthetic input fixtures, including an attenuated case that exercises rejection. Selector outputs are also compared on saved development inputs. The observed receiver point differences in these bounded comparisons are zero.

Installation checks exercise CPU commands, clean source and wheel payloads, and an installed neural receiver through prepared and sample-paced interfaces. The dependency environment is the tested existing stack. Full dependency installation, complete waveform hash recovery, full cohort synthesis, retraining, recalibration and a new timing campaign are not part of these checks. The four-second scheduling path is included, but the bounded installation smoke uses a one-second recording.
