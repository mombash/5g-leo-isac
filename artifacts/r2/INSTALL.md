# Reproducible LEO OFDM implementation

This package implements the selected low Earth orbit (LEO) orthogonal frequency division multiplexing (OFDM) simulator components, statistical/neural query models and eight receiver configurations. The standalone Python namespace is **`leo_ofdm_repro`**. The r2 candidate includes raw generation, the sample-paced frontend/scheduler, model execution and all eight selected receiver routes. It does not require the original checkout, historical experiment runners or machine-specific data links after installation.

The implementation is derived from the current committed source versions:

| Source | Current local commit |
|---|---|
| Simulator, physical search, refinement and model definitions | `194f43575a482fcd316a78e5805a6786462bfcb2` |
| Evaluated controller, scoring and configuration adapters | `495f4f757464f8edda0a1d4fe457e74c0e5f232d` |

The first source repository was already clean. The second commit records 33 implementation files at their existing location. Generated results/data and 24 unrelated untracked entries were left untouched. The older package commit `803f86f` is scaffolding, not the authority for source selection. The current source-content snapshot is `74d685d6bd035144c6aaf67afc9f900c2247323090c5162c6b13517136668cc5`; inspected current scientific/adaptor identities match the evaluated identities. Function-level mapping and source commits are included in the wheel.

## Install and run

The internal delivery directory contains the versioned wheel, a separate selected-assets ZIP, notices and a hash manifest. The seven scientific asset files total approximately 2.2 MiB: the two selected checkpoints, their model specification and shared normalization, the rate prior, final gate and physical configuration. The ZIP also contains its verification manifest. Recipients do **not** need to reconstruct these assets from the original server.

From an unpacked delivery directory:

```sh
python3 -m pip install ./leo_ofdm_reproducible_package-0.1.0-py3-none-any.whl
python3 -m zipfile -e leo_ofdm_selected_assets-0.1.0.zip .
python3 -m leo_ofdm_repro methods
python3 -m leo_ofdm_repro example
python3 -m leo_ofdm_repro verify-assets --assets ./selected_assets
```

The wheel's CPU dependencies are NumPy 2.1.3 and SciPy 1.15.3. Receiver execution additionally uses the declared `gpu` extra: PyTorch 2.6.0+cu124 and Triton 3.2.0 on a compatible NVIDIA/CUDA setup. See [running instructions](RUNNING.md) for the exact GPU installation, input, generation and build recipes. The evaluation environment was not upgraded or changed to build the package.

Fresh CPU and complete GPU dependency stacks were downloaded and installed into separate environments. Installed CPU/generation commands and all eight paced GPU routes pass with reads from original and package-authoring source trees denied. The complete tested dependency versions are supplied in `requirements-cpu.lock` and `requirements-gpu.lock`. The build backend is pinned separately to setuptools 75.8.0 and wheel 0.45.1.

## Raw sample-paced execution

```sh
python3 -m leo_ofdm_repro paced --assets ./selected_assets --method N5 \
  --reference ./raw/reference.npy --observation ./raw/observation.npy \
  --alignment ./raw/alignment.npy --output ./checks/N5-paced.json
```

Inputs are one or four seconds of complex64 raw samples at 30.72 MHz, with a preceding 10 ms reference guard and one `[delay_samples,delay_rate_samples_per_s]` alignment row per 20 ms. The one-second route uses the committed short-recording scheduler adapter; four seconds retains the original four-attempt scheduler. The CLI applies the committed warmup sequence before the measured origin. It records actual preparation/search/emission times and checks causality. It does not estimate alignment or reproduce RF acquisition. `timing_valid=true` identifies this digital boundary, not matched paper latency. See [running instructions](RUNNING.md) for complete generation and waveform recipes.

## Selected configurations

| ID | What executes |
|---|---|
| N5 | Three paid initial motion searches, then two neural-selected searches at each early check; physical alert and continued final refinement |
| S5 | The same five-search structure using the frozen statistical selector |
| C9 | Prior-ordered nine-search classical early checks with the same physical detector/refinement |
| E27 | Exhaustive 27-rate native search at one second |
| DS | Adapted dual-scale full-rate search with common physical refinement |
| TS1 | Two-step detector family's selected fixed-one-look operating point |
| TS2 | Its separate fresh-two-look operating point; unresolved cells retain identity across the new suffix |
| KT+LVD | Selected OLD full-density keystone-transform/Lv-distribution support adaptation |

The retained historical thresholds are the selected operating point. **Fresh calibration is not pending or silently performed.** TS2 preserves the evaluated adapter's upward float32 conversion of its first-look lower/upper thresholds. KT+LVD uses source SHA-256 `d0a39b0894238572c66b23e36cd6a9912377ffbe4ddb628146b69fa9127fe5b7`, 4000 lags/scaled-time positions, 32-delay batches, the matrix transform and 256-lag blocks. A newer source-matched candidate is not substituted.

## What is verified

The runtime passes 13 behavioral/asset tests plus seven raw-input/causality tests, 40 tiny scoring examples from completed saved outputs, current-scene geometry and 200 native interpolation comparisons, and one-input GPU comparisons for **each of the eight receiver routes**. The new paced comparison includes actual cancellation/correlation/preparation, scheduling and measured output availability. Alert decisions, candidate choices, physical estimates and applicable continuation-gate outputs match the current committed implementations; maximum observed point difference was zero. Actual three-frame NR I/Q generation passes; a 20 ms propagated raw slice matches the current committed class bit for bit for reference, surveillance and alignment. The short waveform has its own normalization; full-waveform identity is not claimed.

These checks validate their stated cases, not new detection probabilities or latency advantages. The 17 tests and 288 method-attempt replays from the earlier authoring namespace remain separately labeled historical evidence; they are not used as automatic proof for this new runtime.

`prepared` consumes exactly one second of **already prepared** complex correlations/reference variances. It returns `timing_valid=false`: it does not recreate sample collection, the shared raw frontend's elapsed time, queueing or sample-to-alert latency. The separate `paced` command executes the raw frontend and original scheduler with measured timestamps; its boundary is host-resident digital samples, excluding disk/RF/ADC/acquisition/alignment estimation and initialization/warmup. Full four-second recording generation, the four-attempt concurrency run and matched paper-performance reproduction were not exercised in this bounded work. Existing completed campaign results retain their own source/configuration/timing provenance.

The installed simulator includes the original geometry/noise/recording-generation operations, a portable layout adapter, the committed one-second recording adapter, the native interpolation source and the n256 MATLAB builder/helpers. The `simulation-example` command exports a recorded scene/seed example; `geometry`, `generation-plan` and `generate` provide portable inputs, planning and explicit raw synthesis. Full generation can produce large files; installation and scoring examples do not invoke it. No campaign, retraining, sweep or large dataset was generated during packaging; owned bounded smoke inputs remain separate from evaluated results.

## Evidence and scientific limits

The completed evidence records 256 target recordings (188 new plus 68 retained), 300 new one-second empty recordings and 16 older four-second empty recordings retained with their provenance. These are evidence descriptors, not default sampler sizes or a requirement to rerun the campaign. No performance table is copied into this package, and final paper-facing inclusion/wording remains provisional.

Offline scoring preserves **correct alert at any time**, **final correctness**, **wrong first positive alert**, and **any empty-scene false alert** as distinct outcomes. Later correct refinement does not erase an earlier wrong alert; later wrong output does not erase an earlier correct alert. Time to first correct alert is conditional on success; misses retain terminal time only in the defined operational summary. See [source/validation checkpoint](VALIDATION.md) and [source map](SOURCE_MAP.md).

Required credits, declared baseline adaptations and source-recreation uncertainties are retained in [THIRD_PARTY_NOTICES.md](NOTICES.md) and the wheel's `NOTICES.md`. Public distribution rights remain unresolved; this is a local internal delivery, with no push or publication. There is no author-code reproduction, neural-superiority, global classical speed-limit, equal-optimization-effort or complete deployed-receiver latency claim.

The older `src/leo_ofdm` authoring bridge, legacy vendor snapshots, internal provenance/review material and historical experiment runners remain available for development traceability. They are excluded from the wheel and source distribution. The primary user commands are the `leo_ofdm_repro` commands above.
