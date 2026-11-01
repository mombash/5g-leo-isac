# Installation and standalone reproduction

Use `leo_ofdm_repro`. The r1 prepared-input handoff remains immutable. r2 adds the current committed sample-paced scheduler, raw generation commands, a recorded scene/seed example and fresh dependency-installation checks. Original source/results/manuscript are unchanged.

## Install the local delivery

This repository includes the unchanged r2 delivery under artifacts/r2 and its extracted companion bundle under selected_assets. Run these commands from the repository root. No original server checkout is required. Read the root README model-version warning before interpreting reproduction claims.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.lock
.venv/bin/python -m pip install --no-deps .
.venv/bin/python -m leo_ofdm_repro methods
.venv/bin/python -m leo_ofdm_repro example
.venv/bin/python -m leo_ofdm_repro verify-assets --assets ./selected_assets
```

For GPU execution, use a separate environment and install `requirements-gpu.lock` instead of the CPU lock, with the CUDA 12.4 PyTorch index:

```sh
.venv/bin/python -m pip install -r requirements-gpu.lock \
  --extra-index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install --no-deps .
```

The wheel also declares the equivalent top-level `gpu` extra (`torch==2.6.0+cu124`, `triton==3.2.0`). The locks record the complete dependency versions installed during this check. CPU runtime pins are NumPy 2.1.3 and SciPy 1.15.3. The tested interpreter is Linux/Python 3.10.12; GPU execution used an NVIDIA A10-24Q. Build tools are separate: setuptools 75.8.0 and wheel 0.45.1. The evaluated environment was not modified.

Fresh CPU and complete GPU dependency stacks were downloaded and installed into isolated environments. The installed CPU commands, raw-generation slice and all eight installed paced receiver routes were exercised outside the checkouts with original-source and package-authoring-source reads denied. This is a bounded installation/execution check, not cross-platform certification. The GPU environment occupied about 5.2 GiB; allow additional space for downloads, temporary extraction and caches.

The companion ZIP contains seven model/policy/configuration files plus its manifest. Keep it read-only. Verification checks the complete expected file/hash map and actual bytes; empty and tampered bundles are rejected. Retained thresholds and TS2 float32 rounding remain fixed; no calibration is performed by these commands.

## Sample-paced raw receiver

```sh
.venv/bin/python -m leo_ofdm_repro paced \
  --assets ./selected_assets --method N5 \
  --reference ./raw/reference.npy --observation ./raw/observation.npy \
  --alignment ./raw/alignment.npy --output ./checks/N5-paced.json
```

Select N5/S5/C9/E27/DS/TS1/TS2/KT+LVD. Supply NumPy arrays:

| Input | Required form |
|---|---|
| Reference | One-dimensional complex64, `round((duration_s+0.01)*30.72e6)` samples; includes the preceding 10 ms guard |
| Surveillance observation | One-dimensional complex64, `round(duration_s*30.72e6)` samples |
| Alignment | Floating array `(duration_s*50,2)` with `[delay_samples, delay_rate_samples_per_s]` per 20 ms block |

Supported durations are exactly 1 s and 4 s. Four seconds uses the original four-attempt scheduler. One second uses the committed capacity/controller-count adapter used for the completed short-recording evaluation. Every attempt retains its own one-second observation clock. KT preserves its one-search-worker scheduler; the other selected routes retain two search workers and their original controller streams. N5/S5/C9 retain early emitted warnings and the final continuation/refinement output separately.

The command loads raw files, initializes the receiver and executes the committed warmup sequence before the measured clock starts. Warmup prepares the first supplied second and calls the selected physical shapes; a native-query backend supplies the committed persistent-search-stream warmup for standalone baseline runs. It does not use truth or adapt parameters. The warmup source/input are explicit in the output. The scheduler then waits for each digital sample boundary, executes cancellation, correlation and reference-variance preparation on its preparation stream, queues actual searches and records measured frontend/search/output times. It checks that no preparation/search/output precedes the required samples or preparation. No truth or target identity enters the receiver. Outputs are not reconstructed from prepared results or assigned synthetic availability times.

`timing_valid=true` means actual timestamps for this stated **host-resident digital sample-paced boundary**. Disk reads, initialization/JIT/model loading, RF/ADC, acquisition and alignment estimation are excluded. Warmup, hardware, contention and supplied inputs affect elapsed times. The smoke receipts establish causal execution and numerical agreement on one synthetic input per method; they do not establish matched paper latency or performance. The default four-second concurrency path is source-extracted but was not executed in this bounded check.

## Prepared receiver and offline scoring

```sh
.venv/bin/python -m leo_ofdm_repro prepared \
  --assets ./selected_assets --method N5 \
  --correlation ./one_attempt/correlation.npy --variance ./one_attempt/variance5.npy \
  --output ./checks/N5-prepared.json
```

Prepared inputs are exactly `(20000,1792)` complex64 correlations and `(200,1792)` float32 reference variances, spanning one second. Delay is in samples, Doppler in Hz and rate in Hz/s. The sample rate is 30.72 MHz; occupied bandwidth is 18.36 MHz. This command retains `timing_valid=false`; its samples are already prepared.

`example` checks 40 tiny saved-attempt scoring examples without new inference. To score actual paced output offline:

```python
from leo_ofdm_repro.scoring import events_from_attempt, score_emissions
result = score_emissions(events_from_attempt(attempt), present=True,
                         truth_at=endpoint_truth, terminal_s=attempt['decision_s'])
```

`first_correct_available_s`, miss, final correctness, wrong first positive alert, any empty false alert and final false alert remain distinct. Later correction does not erase a wrong first alert; a wrong final output does not erase earlier correctness. Conditional first-correct times and terminal-time operational summaries retain their source definitions. Prepared results must not be assigned invented timestamps for this scorer.

## Recorded example, geometry and recording generation

The installed package supplies one existing recorded scene and its waveform/noise seeds as an input example. It is not a new evaluation cohort or a new default sampler.

```sh
.venv/bin/python -m leo_ofdm_repro simulation-example \
  --waveform "$PWD/waveform.c64" --output ./inputs
.venv/bin/python -m leo_ofdm_repro generation-plan \
  --manifest ./inputs/paper_consolidated_2026_10_28_test_continued_v5_manifest.json
.venv/bin/python -m leo_ofdm_repro geometry \
  --assets ./selected_assets --scene ./inputs/scene.json \
  --output ./inputs/geometry/paper_consolidated_2026_10_28_test_continued_v5/paper_consolidated_2026_10_28_test_continued_v5_path_000.npz
```

`simulation-example` reports the exact split, episode and scene IDs used below. Geometry writes its `.npz` arrays and separate `.json` summary. Endpoint truth stays outside inference. For other recorded inputs, retain scene fields, seeds, split/parent grouping, waveform identity, geometry and reference choice. Relative waveform cache paths resolve under the manifest directory. Existing sealed/approved-reference guards remain active.

Build the shipped native interpolation source explicitly, using the installed package location:

```sh
g++ -O3 -fPIC -shared -fopenmp /path/to/installed/leo_ofdm_repro/native/kernels.cpp -o ./kernels.so
export LEO_OFDM_KERNEL_LIBRARY="$PWD/kernels.so"
```

The package-owned g++ 11.4.0 build matched the retained library hash and 200 interpolation outputs. No machine binary is shipped.

After obtaining the waveform as described below, a bounded 20 ms raw smoke is:

```sh
.venv/bin/python -m leo_ofdm_repro generate \
  --assets ./selected_assets --manifest-directory ./inputs \
  --geometry-root ./inputs/geometry --contract ./inputs/contract.json \
  --scratch-root ./scratch --split paper_consolidated_2026_10_28_test_continued_v5 \
  --episode-id paper_consolidated_2026_10_28_test_continued_v5_path_000_draw0_rcs0.10 \
  --smoke-frames 1 --output ./raw-slice
```

The smoke limits only output lengths/alignment centres to 1–5 twenty-ms frames, preserving propagation/interpolation/noise operations. A 20 ms reference/observation/alignment slice matched the current committed class bit for bit on the same supplied short NR waveform. A slice is too short for the receiver's one-second attempt.

For a complete recording, omit `--smoke-frames` and supply `--duration 1` or `--duration 4`, using a sufficiently long waveform. Those original recording APIs are included; full recording generation was not run here. Raw arrays alone occupy approximately 494 MB for one second or 1.97 GB for four seconds; clean component caches, waveform storage and transient arrays need additional space. Generation is explicit and never invoked by installation or examples. No campaign/training/sweep is included.

## NR waveform generation

MATLAB R2026a with the required 5G Toolbox functions is the tested configuration. Add the installed `leo_ofdm_repro/matlab` directory to MATLAB's path and use `maxNumCompThreads(4)`.

For the bounded smoke, supply the exported `inputs/waveform.json`:

```matlab
result=generate_waveform_smoke('inputs/waveform.json','waveform.c64',3);
```

Three NR frames produce 30 ms (921600 complex samples, about 7.4 MB), sufficient for the 20 ms raw slice plus reference guard. The smoke uses the recorded cell/payload/control seeds and n256/full-occupancy helpers. It normalizes over the short waveform only; it does not reproduce the full waveform's normalization/hash. Actual I/Q generation passed. A separate feature-name license-test query returned false while the waveform functions executed successfully; the runtime also emitted a two-day expiry notice at the check. Neither result is treated as proof that the waveform functionality is unavailable.

For the unchanged full recipe:

```matlab
prepare_manuscript_waveforms('inputs','paper_consolidated_2026_10_28_test_continued_v5',1);
```

It retains 402 frames, recorded seeds, full-waveform normalization and interleaved little-endian complex float32 output. Only packaged helper-path resolution differs from the current source. One 4.02 s waveform is about 988 MB. Full generation and retained waveform-hash recovery were not checked in this bounded work.

## Validation and development

The candidate records 13 original standalone behavioral/asset tests plus seven raw-input/causality tests; 40 saved scoring examples; current geometry/native comparisons; actual short NR/raw generation; eight current-source paced route comparisons; and installed fresh CPU/GPU checks. Each receipt states its inputs and scope. Older authoring-namespace checks remain historical.

Build with the separately pinned backend:

```sh
python3 -m pip wheel --no-deps . --wheel-dir ./dist
```

The delivery manifest binds source commits, wheel payload, companion assets and documentation. Authoring provenance/tools and the old vendor tree are excluded from the wheel/source distribution. No original scientific source/results/manuscript are edited, and no push/publication is performed. Public redistribution terms remain a separate unresolved release matter.
