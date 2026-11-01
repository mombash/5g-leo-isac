# r2 standalone simulator/receiver checkpoint — 28 October 2026

r2 extends the immutable prepared-input candidate `fb438952a246a89ed7ea5d78d12bcf517154cc06` with portable generation inputs/commands and the original digital sample-paced frontend/scheduler/controllers for all eight selected routes. r1's delivery remains unchanged. A later package commit and exact delivery/file manifests identify r2.

## Source and scientific identity

Source commit: `194f43575a482fcd316a78e5805a6786462bfcb2`. Evaluated adapter/scoring commit: `495f4f757464f8edda0a1d4fe457e74c0e5f232d`. Source snapshot: `74d685d6bd035144c6aaf67afc9f900c2247323090c5162c6b13517136668cc5`. The first repository remains clean; the second retains its 24 unrelated untracked entries. No original scientific source/results/manuscript changes are part of r2.

Retained-frozen-v5 thresholds, models, feature/scoring definitions, TS2 upward float32 bounds and OLD KT+LVD selection remain fixed. KT retains nt4000/batch32/matrix/lag_block256 and one search worker. No calibration, training, sweep, dataset expansion or baseline substitution occurs.

## Implemented and bounded checks

| Capability | Evidence and limits |
|---|---|
| Fresh CPU dependencies | NumPy 2.1.3/SciPy 1.15.3 downloaded and installed into a venv without host site packages; installed methods/examples/assets/recorded-input/geometry/raw-slice commands run with source reads denied. |
| Fresh GPU dependencies | Complete PyTorch 2.6.0+cu124/Triton 3.2.0 and declared dependency stack downloaded into a separate venv on owned large-volume scratch. All eight installed paced routes run; outputs match prior current-source checks. Full tested versions are supplied as locks. Evaluated environment unchanged. |
| NR waveform | Actual 3-frame, 30 ms n256/full-occupancy waveform generated with recorded seeds: 921600 complex samples, about 7.4 MB. Short-output normalization only. No 402-frame/hash recovery claim. |
| Raw generation | One existing target scene/seeds through a 20 ms propagation/interpolation/noise slice. Reference, observation and alignment match current committed class bit for bit on the same supplied short waveform. Portable example/geometry/generate commands also run from the installed CPU wheel. |
| Paced receiver | One synthetic 1 s NR-based direct-path/echo/noise raw fixture through all eight original controllers. Actual cancellation/correlation/reference variance preparation, queueing, physical searches and emitted outputs execute. Prepared arrays match current source bit for bit; predictions/alerts/selected candidates match, maximum point difference 0. Actual timestamp causality checked. |
| Warmup | Committed campaign warm_backends sequence prepares the first supplied second and selected physical shapes before the measured origin. Single-baseline execution also supplies the original native-query search-stream warmup. Warmup inputs/source are explicit. No truth/parameter adaptation. |
| Behavioral contracts | 13 standalone scoring/asset tests plus seven raw-input/causality tests. Forty saved scoring examples retain wrong-first/later-correct, earlier-correct/wrong-final and any-empty-false-alert definitions. Historical 17 authoring tests/288 replays remain separate. |
| Earlier numerical checks | One registered geometry scene (8 arrays/4041 time points), 200 native interpolation values and all eight prepared routes match current committed sources for their stated cases. |

Actual NR functions executed successfully under MATLAB R2026a. A separate feature-name license-test query returned false despite successful generation; the runtime emitted a two-day expiry notice. Those facts are retained without inferring that functionality is unavailable. No license acquisition/renewal occurred.

The first paced reference check exposed an omitted KT scheduler policy in new wrapper wiring; restoring the exact committed looks=[0]/k=53 fixed it. The failed owned validation process was stopped after it retained worker/GPU resources. Wrapper error cleanup now stops failed workers. These are packaging fixes; successful scientific arithmetic/control policies remain unchanged.

## What the interfaces reproduce

`paced` accepts exactly 1 s or 4 s raw complex64 samples, the reference's preceding 10 ms guard and supplied 20 ms alignment rows. Four seconds retains the original four-attempt scheduler; one second applies the committed capacity/controller-count adapter. It reports actual frontend/search/output times, checks causality and declares its digital boundary. `prepared` remains a one-second quality interface with timing_valid=false. Offline truth/scoring stays outside inference.

The timing boundary starts with host-resident digital samples, after initialization/warmup. Disk reads, RF/ADC, beam acquisition and alignment estimation remain excluded, as in the selected source setup. A timing_valid=true output reports actual execution at this boundary; it is not a matched paper latency/performance claim. Host/load/warmup/input differences remain material.

Full one-/four-second raw generation and the unchanged 402-frame waveform recipe are included but were not run in this bounded work. The four-attempt concurrency route is source-extracted but was not executed here. The actual paced smoke uses a repeated short NR waveform plus deterministic direct/echo/noise components, not a recovered evaluated physical recording. No detection probabilities or paper result tables are inferred from it.

## Evidence and handoff

Completed campaign evidence still identifies 256 targets (188 new plus 68 retained), 300 new one-second empties and 16 older four-second empties with their provenance. No saved result pooling/rewriting or manuscript wording change occurs.

The local handoff contains the standalone wheel, the same verified seven-file companion bundle, installation/generation/receiver recipes, dependency locks, credits and exact hashes. Private source paths, original vendor trees, internal discussions and raw/generated data are excluded. Public redistribution terms remain separate; nothing was pushed or published. The existing reviewer can assess the changed readiness evidence against the final candidate, without a new scientific audit.
