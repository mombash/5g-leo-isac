# Passive UAV sensing with 5G NR LEO downlinks

This repository implements the receiver and simulator studied in *Early UAV Detection Using 5G NR LEO Downlinks for Passive ISAC*. A ground receiver uses a 5G New Radio (NR) satellite downlink as an illuminator and detects an unmanned aerial vehicle (UAV) from its reflected signal. The code includes classical, statistical and neural search selection, five comparison receiver configurations, trained selector models, and compact stored evaluation outputs.

The receiver takes an aligned reference signal and a surveillance signal. It suppresses direct-signal leakage, computes correlations, searches candidate rates of Doppler change, and refines delay and Doppler estimates. Our proposed adaptive classical, statistical and neural configurations check at 0.6, 0.8 and 1 second. An alert at an early check can be updated or withdrawn at the final check. Statistical and neural models choose which rates to search. Detection itself uses the refined correlation score.

## Requirements by task

| Task | Requirements |
|---|---|
| Aggregate stored results, export metadata, compute geometry | Linux, Python 3.10, NumPy and SciPy. |
| Generate NR waveforms | MATLAB with working 5G Toolbox waveform functions. |
| Synthesize received samples from waveforms | CPU Python environment plus g++ with OpenMP. |
| Run any receiver, including classical methods | NVIDIA CUDA GPU, PyTorch and Triton. |

## Start with the stored results

These commands require a Git checkout, Linux and Python 3.10. They install the CPU dependencies and recompute detection and recorded timing statistics from the included outputs:

```sh
git clone https://github.com/mombash/5g-leo-isac.git
cd 5g-leo-isac
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.lock
.venv/bin/python -m pip install --no-deps .
.venv/bin/python -m fiveg_leo_isac verify-assets --assets selected_assets
.venv/bin/python -m fiveg_leo_isac results
```

Asset verification should report `passed: true` and seven files. The results command returns JSON keyed by descriptive method names. For example, `quality.adaptive-neural.final_correct` is **1903** out of **2048** attempts, and `false_alerts.adaptive-neural.any_alert_count` is **3** out of **3000** recordings. Timing fields ending in `_s` are seconds. `--no-bootstrap` skips confidence-interval recomputation.

This example aggregates saved predictions and timestamps. It does not generate radio waveforms, simulate received signals, run model inference or measure new processing times. See [reproducibility and metrics](docs/REPRODUCIBILITY.md) for the data populations and statistical definitions.

The installed `.venv/bin/5g-leo-isac` command accepts the same subcommands as `.venv/bin/python -m fiveg_leo_isac`.

## Choose a receiver

Run `.venv/bin/python -m fiveg_leo_isac methods` to list the configurations. Use these names with `--method`:

| Name | Role | Operation |
|---|---|---|
| `adaptive-neural` | Proposed receiver | Search three initial Doppler rates, use the neural selector to choose two more, then refine the strongest peak. |
| `adaptive-statistical` | Proposed receiver | Use the fitted statistical selector for the same three-plus-two searches. |
| `adaptive-classical` | Proposed receiver | Search nine fixed rates at each early check and refine the strongest peak. |
| `exhaustive-classical` | Full-search reference | Search all 27 rates after one second. |
| `dual-scale` | Literature comparison | Reuse coarse migration correction across fine rate searches. |
| `two-step-fixed` | Literature comparison | Two-step detector configured for a single one-second check. |
| `two-step-early` | Literature comparison | Check at 0.8 seconds and test unresolved cells on the next 0.2 seconds. |
| `hybrid-integration` | Literature comparison | Apply keystone alignment and Lv’s distribution to one second of NR observations. |

The three selective receivers share the final search rule. An earlier alert requires all 27 rates at one second. Without an earlier alert, the receiver searches 15 rates from a fixed prior and may stop with no alert. Otherwise it searches the remaining 12 rates and refines the strongest peak. Optional `*-full-final` controls disable the 15-rate rejection shortcut. The two-step and hybrid methods have their own fixed detector settings. Method credits and adaptations are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Run on raw signals

Receiver execution requires Linux, an NVIDIA CUDA GPU, PyTorch and Triton. The tested setup uses Python 3.10.12, PyTorch 2.6.0 with CUDA 12.4, Triton 3.2.0 and an NVIDIA A10-24Q with 24 GB memory. Use a separate GPU environment:

```sh
python3 -m venv .venv-gpu
.venv-gpu/bin/python -m pip install -r requirements-gpu.lock \
  --extra-index-url https://download.pytorch.org/whl/cu124
.venv-gpu/bin/python -m pip install --no-deps .
```

Given a one- or four-second recording in the [documented array format](docs/USAGE.md#raw-input-format):

```sh
.venv-gpu/bin/python -m fiveg_leo_isac paced \
  --assets selected_assets --method adaptive-neural \
  --reference work/raw/reference.npy \
  --observation work/raw/observation.npy \
  --alignment work/raw/alignment.npy \
  --output work/neural.json
```

The output contains alerts, delay/Doppler estimates, searched candidates, and measured output-availability times. The clock starts after loading and warmup, when host-resident digital samples begin to be released at acquisition rate. It includes preparation, GPU transfers, searches, inference, refinement and waiting. Radio reception, analog-to-digital conversion, beam/reference acquisition and disk reads are outside this boundary.

Raw recordings and NR waveforms are not bundled. [Usage instructions](docs/USAGE.md) give a complete path from the included scene and seeds to geometry, MATLAB waveform generation, received-signal synthesis and receiver execution. They also describe prepared-correlation input and output units. Full waveform generation requires **MATLAB with 5G Toolbox**. Received-signal generation requires a **C++ compiler with OpenMP**. Stored-result aggregation requires neither.

## Models and data

`selected_assets/` contains the final neural and statistical selectors, feature normalization, the rate prior, gate settings and physical configuration. `verify-assets` checks the complete file/hash map before use. Model hashes and interpretation are listed in [reproducibility](docs/REPRODUCIBILITY.md#models-and-fixed-settings). The models select searches, not target labels.

The stored quality outputs cover 512 independent simulated four-second UAV recordings and 2048 dependent one-second attempts. The false-alert outputs cover a separate 3000 one-second no-UAV recordings. Recorded timing comes from 16 UAV and eight no-UAV four-second recordings. Results are conditional on the fixed library of nineteen NR waveforms and the modeled line-of-sight scene, acquired beams and supplied reference alignment.

## Repository layout

| Path | Contents |
|---|---|
| `src/fiveg_leo_isac/` | Receiver, simulator, scoring, commands, compressed results and scene/seed metadata. |
| `selected_assets/` | Trained models and fixed receiver/physical settings, with SHA-256 hashes. |
| `docs/USAGE.md` | Input generation, inference interfaces and output interpretation. |
| `docs/REPRODUCIBILITY.md` | Data units, endpoints, timing, uncertainty and model identities. |
| `tests/` | Scoring, asset integrity and sample/timestamp causality checks. |

The implementation has bounded numerical comparisons against its evaluated receiver code and tests for scoring, assets and input causality. Source and wheel installations are checked. These checks do not establish full dataset regeneration or repeatable timing on other hardware. Development and build commands are in [usage](docs/USAGE.md#development).

See [RIGHTS.md](RIGHTS.md) for distribution terms and external dependencies.
