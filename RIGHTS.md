# Rights and dependencies

This repository provides public access to the research implementation and the selected frozen research-model weights. No separate open-source license has been selected or added. Retain the scientific credits and adaptation notices in THIRD_PARTY_NOTICES.md.

The MATLAB files are research configuration and generation helpers that call MATLAB/5G Toolbox APIs. MATLAB, 5G Toolbox, their implementations, installers, licenses and generated waveform data are not distributed here. Obtain external Python/CUDA/MATLAB dependencies under their own terms. No third-party library binary or source distribution is vendored.

The final neural asset contains model tensors only, exported without optimizer/RNG/training metadata. Its original checkpoint hash and exported-file hash are recorded separately. S5 and the selected normalization/prior remain frozen. Historical r2 artifacts and notices are labeled as such; they do not define the submission defaults.

See provenance/ORIGINAL_THIRD_PARTY_NOTICES.md for the preserved original attribution record. Its historical references to an authoring/vendor tree describe the original local package; that tree was never exported into this repository.
