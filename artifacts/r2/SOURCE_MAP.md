# Current committed source map

The standalone runtime is derived from source commit `194f43575a482fcd316a78e5805a6786462bfcb2` and evaluated adapter/scoring commit `495f4f757464f8edda0a1d4fe457e74c0e5f232d`. The current source-content snapshot is `74d685d6bd035144c6aaf67afc9f900c2247323090c5162c6b13517136668cc5`. Extraction reads committed bytes and refuses concurrent working-file changes. Original source/results/manuscript remain unchanged.

`leo_ofdm_repro/_config/extraction.json` records 92 definitions across 42 generated modules, source paths/hashes, retained members and numerical AST identities. Portable wiring and CLI files are package-owned. Comments and unused runners are excluded; notices retain credits and baseline-adaptation qualifications. The older vendor/authoring tree is historical and excluded from distributions.

| Package modules | Current source behavior |
|---|---|
| scenes, geometry, noise, simulation_data, interpolation | Recorded geometry/steering, propagation, complex noise, native interpolation and raw recording class |
| frontend, cancellation, warp, warp_table, decimation | Sinc direct-path cancellation, correlations/reference variances and motion search operators |
| coherent, native, verifier, refinement | Dense physical search and unchanged local/phase refinement |
| physical_features, set_features, phase_features, innovation, evidence, measurements | Actual paid evidence and selected model inputs |
| model_core, model_static, reader_core, query_reader | Selected neural and statistical readout, including captured GPU execution |
| base_backend, final_backend, query_stage, gate | E27 and N5/S5/C9 physical stages and final continuation gate |
| dual_scale, ds_backend | Selected all-rate dual-scale adaptation |
| two_look, ts_backend, ts_scheduler | Selected TS1/TS2 physical callbacks and original controller |
| kt_product, kt_lvd, kt_backend, kt_scheduler | Selected OLD full-density KT+LVD and isolated one-look timing namespace |
| scheduler, query_scheduler | Original preparation/dispatch/search/controller execution, including the evaluated query-controller adapter |
| duration_adapters | Committed one-second recording and scheduler capacity/controller-count adapters |
| warmup | Committed campaign warm_backends preparation and physical-call sequence, before measured origins |
| scoring, numerics | Current emitted-event scoring/aggregation and upward float32 conversion |

`simulation.py` supplies portable layout/path wiring, explicit 1 s/4 s APIs, a 1–5-frame sliced smoke and a recorded input example. The smoke changes output lengths only and is not a new scientific sampler. The full waveform builder retains 402 frames/seeds/normalization; its only change is adjacent helper-path resolution. The separate short waveform helper declares its sliced normalization and is not a recovered evaluated waveform.

`paced.py` selects original backends/controllers, applies the committed one-second adapter when requested, retains KT's single search worker, runs the committed warmup sequence, checks raw input requirements and verifies measured causality. No truth enters the receiver. Errors trigger worker cleanup; successful-path scheduler arithmetic/timestamps are source-derived. `prepared` remains a distinct untimed-quality interface.

The seven companion assets are bound by complete expected hashes. Thresholds remain retained-frozen-v5; TS2 preserves upward float32 first-look bounds. OLD KT+LVD source hash is `d0a39b0894238572c66b23e36cd6a9912377ffbe4ddb628146b69fa9127fe5b7`, nt4000/batch32/matrix/lag_block256. No baseline substitution, calibration, retraining or scientific correction occurs.

Receipts distinguish original prepared checks, new raw/paced comparisons, fresh installed environments and sliced waveform generation. One-input checks do not establish detection probabilities or matched paper latency. Full 402-frame generation and the four-attempt concurrency run remain unexecuted here.
