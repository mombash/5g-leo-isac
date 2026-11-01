# Running the simulator and receiver

Run commands from the repository root after the CPU or GPU installation in the [README](../README.md). Commands write to new paths and preserve existing outputs. The examples use `work/` for generated files. Choose a location with sufficient free space before generating full recordings.

## Inspect stored results and input cohorts

```sh
.venv/bin/python -m leo_ofdm_repro results
.venv/bin/python -m leo_ofdm_repro example
.venv/bin/python -m leo_ofdm_repro cohorts
```

`results` reaggregates saved detections and timestamps, including paired bootstrap confidence intervals. `example` checks the alert scorer on 40 small saved examples. Neither runs a receiver. `cohorts` lists the scene/seed metadata that can be exported for explicit regeneration.

## Generate simulation input

The following sequence starts with one supplied UAV scene. It computes geometry, generates an NR waveform in MATLAB, then synthesizes reference and surveillance signals. No random scene selection is performed by the export command.

### 1. Export a scene and compute geometry

```sh
.venv/bin/python -m leo_ofdm_repro export-inputs \
  --cohort example --waveform-root "$PWD/work/waveforms" --output work/inputs
.venv/bin/python -m leo_ofdm_repro prepare-geometry \
  --assets selected_assets --inputs work/inputs --output work/geometry
```

The input directory contains a manifest, `scene.json`, `waveform.json`, `contract.json` and `inputs.json`. The manifest preserves the scene, noise, payload and control seeds. `inputs.json` identifies the first recording and its geometry paths. Geometry contains bistatic path lengths, antenna fields and endpoint ground truth. Ground truth is used for evaluation and does not enter receiver inference.

For a supplied scene outside this example, `geometry --assets selected_assets --scene scene.json --output geometry.npz` writes both the array file and a JSON summary. `prepare-geometry` processes the **first** exported scene. To regenerate an entire cohort, compute geometry for every scene and run `generate` for each episode ID in its manifest.

### 2. Build the sample interpolator

Received-signal synthesis uses the included C++ interpolation source. Install `g++` with OpenMP support, then run:

```sh
.venv/bin/python -m leo_ofdm_repro build-interpolator --output work/kernels.so
export LEO_OFDM_KERNEL_LIBRARY="$PWD/work/kernels.so"
```

The environment variable must remain set when running `generate`. This library is used for synthesis; receiver inference uses the PyTorch/Triton operators.

### 3. Generate the NR waveform

MATLAB R2026a with working 5G Toolbox waveform functions is the tested configuration. With `matlab` available in the shell:

```sh
mkdir -p work/waveforms
MATLAB_HELPERS="$(.venv/bin/python -m leo_ofdm_repro matlab-path)"
matlab -batch "addpath('$MATLAB_HELPERS'); maxNumCompThreads(4); meta=jsondecode(fileread('work/inputs/inputs.json')); generate_waveforms('work/inputs', meta.split, 1);"
```

The helper reads the first waveform's cell, payload and control seeds from the manifest. It generates 402 NR frames (4.02 seconds), normalizes power over the complete waveform, and writes interleaved little-endian complex float32 samples to the exported waveform path. One waveform occupies about **988 MB**. To generate every waveform listed in a cohort manifest, pass all its one-based waveform indices instead of `1`.

Full waveform hash recovery depends on matching the fixed recipe and MATLAB/toolbox behavior. Expected hashes are retained where available in exported cohort metadata. The repository does not include the nineteen generated waveform files.

For a small diagnostic, replace the full generation command with:

```sh
matlab -batch "addpath('$MATLAB_HELPERS'); maxNumCompThreads(4); generate_waveform_smoke('work/inputs/waveform.json', 'work/waveforms/example.c64', 3);"
```

This writes 30 ms of waveform (about 7.4 MB). It is sufficient for the 20 ms slice below. Its normalization uses only that short waveform, so it will not match the full waveform's hash or yield the paper's results. Choose **one** waveform command per new output directory.

### 4. Synthesize received samples

With a complete waveform, generate the exported four-second recording:

```sh
.venv/bin/python -m leo_ofdm_repro generate \
  --assets selected_assets --manifest-directory work/inputs \
  --geometry-root work/geometry --contract work/inputs/contract.json \
  --scratch-root work/cache --output work/raw
```

`generate` reads the split, first episode ID and duration from exported metadata. It writes `reference.npy`, `observation.npy`, `alignment.npy` and `generation.json`. Use `--episode-id` to select another manifest row. Explicit `--split` and `--duration 1` or `--duration 4` remain available. Keep the manifest's duration for evaluation regeneration.

Raw reference and observation arrays alone occupy about **494 MB for one second** or **1.97 GB for four seconds**. Allow several additional gigabytes for clean-component caches, waveform storage, temporary arrays and GPU preparation. Full waveform and complete cohort regeneration have not been rerun for this distribution.

With the short waveform, use a new output directory and add `--smoke-frames 1`:

```sh
.venv/bin/python -m leo_ofdm_repro generate \
  --assets selected_assets --manifest-directory work/inputs \
  --geometry-root work/geometry --contract work/inputs/contract.json \
  --scratch-root work/slice-cache --smoke-frames 1 --output work/raw-slice
```

This creates only 20 ms of received samples. It tests the generation path and is too short for receiver inference. Use separate cache directories for short slices and full recordings.

## Raw input format

`paced` accepts NumPy `.npy` arrays for exactly one or four seconds:

| Array | Shape and units |
|---|---|
| `reference` | One-dimensional `complex64`, `round((duration_s + 0.01) × 30.72e6)` samples. The extra 10 ms is the preceding reference guard. |
| `observation` | One-dimensional `complex64`, `round(duration_s × 30.72e6)` surveillance samples. |
| `alignment` | Floating array `(duration_s × 50, 2)`. Each 20 ms block supplies `[delay_samples, delay_rate_samples_per_s]` for direct-signal alignment. |

Samples use 30.72 MHz sampling and 18.36 MHz occupied bandwidth. Supplied alignment is a prerequisite. The receiver does not estimate beam or reference acquisition. Simulation generation computes alignment from its geometry.

## Run a sample-paced receiver

Use the GPU environment after producing a full recording:

```sh
.venv-gpu/bin/python -m leo_ofdm_repro paced \
  --assets selected_assets --method neural \
  --reference work/raw/reference.npy --observation work/raw/observation.npy \
  --alignment work/raw/alignment.npy --output work/neural.json
```

Four seconds produces four one-second attempts. Each attempt uses its own observation and output-availability clock, measured from that attempt's first digital sample. Preparation uses one stream. Two search workers run the main receivers, except hybrid integration, which uses one. Later checks can wait behind unfinished searches. Those waits remain in the reported times.

Before the clock starts, the command prepares a second of data, warms the required physical stage shapes and warms both persistent search streams. By default it uses the receiver input for warmup. To use the paper's disjoint calibration warmup protocol, supply all three arrays from a separate calibration recording:

```sh
.venv-gpu/bin/python -m leo_ofdm_repro paced \
  --assets selected_assets --method neural \
  --reference work/raw/reference.npy --observation work/raw/observation.npy \
  --alignment work/raw/alignment.npy \
  --warmup-reference work/calibration/reference.npy \
  --warmup-observation work/calibration/observation.npy \
  --warmup-alignment work/calibration/alignment.npy \
  --output work/neural-calibrated-warmup.json
```

Warmup changes initialization, not thresholds or model weights. `timing_valid: true` means the JSON contains actual timestamps for the documented digital boundary. It does not establish a timing advantage or make times comparable across different hardware, warmup, inputs or load.

## Prepared correlations

If correlations have already been computed, run one attempt directly:

```sh
.venv-gpu/bin/python -m leo_ofdm_repro prepared \
  --assets selected_assets --method neural \
  --correlation work/prepared/correlation.npy --variance work/prepared/variance.npy \
  --output work/prepared-neural.json
```

Inputs are `(20000,1792)` `complex64` correlations and `(200,1792)` `float32` reference variances. The command supports the same receiver methods but skips raw preparation and sample release. Its output therefore has `timing_valid: false` and no measured output-availability times. Prepared arrays are external inputs and are not bundled.

## Read receiver outputs

`paced` returns an `attempts` list plus `shared_stages` and `executed_searches` timing traces. For selective receivers, `attempts[i].updates` contains the emitted checks. Other receivers report the terminal result under `attempts[i].output`. `prepared` returns an `events` list.

| Field | Meaning |
|---|---|
| `alert` | Whether the receiver emitted a target alert at this check. Later withdrawal does not erase an earlier alert. |
| `prediction` | Leading entries are `[score, delay_samples, Doppler_Hz, rate_Hz_per_s]`. Some branches append a search-count field. Score is dimensionless and is not a detection probability. |
| `observation_s` | Duration of samples used by the check, in seconds. |
| `available_s` | Time the emitted update became available, in seconds from the attempt origin. Terminal-only results use `decision_s`. |
| `selected_ids` | Indices into the 27-rate grid `-52, -48, …, 52 Hz/s`, where reported. |
| `executed_rates` | Number of rates searched at this check, where reported. |
| `final_gate_rejected` | Whether the final partial search ended with no alert. |

Delay is excess bistatic echo delay relative to the aligned reference. Convert samples to seconds by dividing by `30.72e6`. Do not interpret estimates attached to a negative alert as detections. [Metric definitions](REPRODUCIBILITY.md#endpoint-definitions) explain how emitted events are scored against truth.

## Development

Run the existing scoring, asset-integrity, raw-input and causality tests from a source checkout with the GPU environment available:

```sh
.venv-gpu/bin/python -m unittest discover -s tests -v
```

Build a wheel into a chosen temporary directory:

```sh
.venv/bin/python -m pip wheel --no-deps . --wheel-dir /path/to/build-output
```

The wheel includes Python code, MATLAB helpers, C++ source and compressed result/input metadata. Trained assets remain in `selected_assets/` and must be supplied to an installed receiver through `--assets`. No compiled interpolation library, generated radio waveform or raw recording is included. Build and test outputs are not part of the repository.
