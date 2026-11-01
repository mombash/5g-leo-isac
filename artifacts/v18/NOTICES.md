# Attribution and rights

This research implementation derives from current source commits:

- Simulator, physical processing and saved query-model definitions: `194f43575a482fcd316a78e5805a6786462bfcb2`.
- Evaluated controller/scoring/configuration adapters: `495f4f757464f8edda0a1d4fe457e74c0e5f232d`.

Function/file hashes and explicit import/wiring adaptations are in `_config/extraction.json`. No author-code reproduction is claimed. No repository distribution license has been established; this package does not invent one. Keep required upstream scientific credits and dependency licenses with any later authorized distribution. No third-party library is vendored here.

Scientific method credits retained from the source records:

- Grossi, Lops and Venturino, “Two-Step Sequential Detection in Agile-Beam Radars: Performance and Tradeoffs,” IEEE TAES 53(5), 2199–2213 (2017), DOI `10.1109/TAES.2017.2688878`. TS1 and TS2 are two operating points of this family, not two unrelated methods. Passive calibration, motion-cell propagation and parameter refinement are explicit adaptations. The second dwell uses fresh samples and preserves unresolved-cell identity.
- Li et al., “Efficient Dual-Scale Generalized Radon-Fourier Transform Detector Family for Long Time Coherent Integration,” IEEE TSP (2024), DOI `10.1109/TSP.2024.3411161`. DS shares migration correction across fine Doppler-rate trials, adapted to bistatic centered-baseband observations with all 27 permitted rates and common local refinement.
- He, Yang and Chen, “A Hybrid Integration Method for Moving Target Detection With GNSS-Based Passive Radar,” IEEE JSTARS 14, 1184–1193 (2021), DOI `10.1109/JSTARS.2020.3037200`. KT+LVD is the selected OLD support adaptation, source SHA-256 `d0a39b0894238572c66b23e36cd6a9912377ffbe4ddb628146b69fa9127fe5b7`. NR frontend, positive-lag finite discretization, one-frame integration and retained detector calibration are disclosed adaptations. It is not the authors' GNSS receiver/code/runtime or a literal implementation of the separate scaling/Fourier resampling route. Finite-boundary/source-recreation uncertainties remain; no newer uncalibrated core is substituted.
- Adaptive information acquisition, neural architectures, statistical regression, interpolation, early checks and classical transforms are established techniques. The original query sources credit Covert et al., ICML 2023, https://proceedings.mlr.press/v202/covert23a/covert23a.pdf. Packaging these operations establishes neither novelty nor a neural-specific advantage.

The simulator uses 5G Toolbox to generate the legal recorded NR waveform identities. MATLAB and Python/CUDA dependencies retain their own licenses. The package omits private discussion, historical training/experiment runners and machine-specific data paths; it retains numerical operations, explicit adaptations, limitations and credits.

Retained thresholds do not establish a fresh calibration or a nominal population false-alarm guarantee. Supplied alignment/beam/reference prerequisites and the digital-sample timing boundary must remain visible. First correct emitted alert, final correctness, wrong initial alert and empty-scene false alerts are distinct measures. No global classical speed limit, equal optimization effort or complete deployed-receiver latency is claimed.
