# Superseded: CWRU arms of round 3

The CWRU configurations of this folder (`e4_bm3_*`, `e5_bm3_*`) were run with the earlier noise
implementation (`noise_key: v1`), which seeded the injected noise by a window's index within its split, so
training, validation and test windows with the same index received the same standard-normal waveform and
the training seed re-paired test windows with noise. They are kept for provenance and are **not used to
support the current manuscript**. The corrected rerun is `results/mst_round4_noisefix_20261007/`.
The XJTU-SY arms of round 3 (`e1_*`, `e2_*`, `e3_*`, `e3t_*`) involve no injected noise and are unaffected.
