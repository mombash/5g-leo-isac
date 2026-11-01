# V18 submission package checkpoint

The default package is version 0.2.0 and uses the final selected V18 models. Original scientific source/campaign commits remain 194f43575a482fcd316a78e5805a6786462bfcb2 and 495f4f757464f8edda0a1d4fe457e74c0e5f232d. The adopted paper source/PDF identities and final timing archive hash are in provenance/SNAPSHOT.json. Manuscript, original package, campaigns and results were not modified.

## Verified for this update

- All 92 extracted scientific definitions match the frozen evaluated source. Normalization, rate prior, gate and physical values match their original prerequisites. Precision, observation prefixes, features, thresholds, refinement, withdrawal/gate rules, scoring and baseline adaptations remain unchanged.
- N5 original checkpoint SHA-256 is 576b78e35a3d966199e588ca3bc9ee2209fe7cebc09023ea40208f93e9bd6be3. The shipped weights-only export SHA-256 is e64a8c0b775c7b4862eb1f426ca0c5945d91061845c63406c481afd8401adeb8. Every model tensor has identical values, dtype and shape. Optimizer/RNG/training metadata is excluded.
- S5 remains byte-identical to the final checkpoint, SHA-256 49b07366253aa2fcdc1f3c6ae08048e0d206ad5165dff7768bc2cba6183d626d.
- Four saved development-input selector comparisons match the final evaluated readers. All eight receivers match predictions/scores, candidate choices, gate decisions and emitted alerts on two bounded saved-input fixtures. The attenuated fixture exercises no-alert/15-rate final rejection; the stronger fixture exercises alerts/full continuation. Maximum receiver point difference was zero.
- The three existing always27 component controls also match the frozen reference on those two fixtures, including 27 final searches on the weak-input case. These are existing evaluated controls, not new methods.
- Twenty standalone scoring/asset/raw-input/causality tests pass.
- Compact saved-outcome recomputation matches all eight quality and fresh-FA endpoints, Nov 1 timing point means and the original 20000-replicate whole-record paired bootstrap. No new inference or measurement is performed by this recomputation.
- Source installation and the built wheel pass methods, full asset verification, saved-outcome recomputation and frozen-input export with original scientific and authoring source reads denied. One installed N5 prepared/paced smoke also matches predictions, candidates, gate decisions and alerts, with actual timestamp causality checked. The explicit warmup-input path is exercised on the smoke input; it is not a recreation of the paper's disjoint calibration warmup record.

The bounded update GPU comparisons use an existing saved synthetic raw direct/echo/noise fixture and its attenuated correlations; they are not recovery of a held-out V18 physical recording. They validate implementation agreement, not detection-rate precision or a timing advantage. Dependency and build environments are the existing validated environments; no full dependency reinstall was performed.

## Reproduction evidence

Quality is exactly 512 independent four-second UAV recordings (sealed original256+new256), giving 2048 dependent attempts. Unsealed extension extras are excluded. False alerts use the separate completed fixed3000 independent one-second no-UAV recordings, counting any emitted alert including withdrawals. Prior300short/16long recordings are not pooled into that endpoint.

Timing uses the final Nov1 contemporaneous panel of 16 UAV and eight no-UAV four-second recordings plus existing component controls. The host-digital boundary includes frontend/preparation, selection/search/refinement, emitted outputs and backlog; disk/RF/ADC/acquisition/alignment estimation/model load/JIT/warmup are excluded. Conditional first-correct and miss-inclusive50/50 operational means remain distinct. Whole-record bootstrap preserves four dependent attempts and uses paired draws across methods. This is recorded-session evidence, not a repeatable wall-clock guarantee or session-variation estimate.

The compact source-result records and frozen scene/seed manifests are included without private server paths. Exported FA metadata is an explicitly opened post-submission reproduction copy; original sealed manifests/results remain untouched. No bulky generated waveforms/raw datasets are shipped.

Full 402-frame waveform/hash recovery, complete physical cohort regeneration, retraining, recalibration and a new timing campaign were not performed. The earlier source/NR/native checks remain historical r2 evidence in R2_CHECKPOINT.md and artifacts/r2. The default V18 wheel and companion asset ZIP are in artifacts/v18; exact hashes and source/install/parity receipts are retained alongside them.
