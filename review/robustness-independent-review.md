# Independent robustness notebook review

Internal reviewer: GPT-6.1 Sol. Review date: 2026-10-31.

## Conclusion

**PASS:** no actionable correctness, protocol, or security findings in the reviewed
implementation, including the final immutable release. This approves a bounded
descriptive robustness replication, not the scientific hypothesis, convergence,
or a compression mechanism. Final runtime identity, launcher readiness, archived
five-cell execution and results presentation all passed independent review.

## Scope and independent checks

Reviewed `AGENTS.md`, baseline audit and relevant training/evaluation machinery,
new production and smoke configs, `docs/robustness.md`, README, builder, generated
notebook, local verifier, and new/config/bootstrap/baseline tests. Confirmed no
numerical source changes from the preceding `31bd42b` release.

- Fresh, non-overlapping seed ranges; only the declared longer student budget,
  wider solo grid and populations change. Teacher settings remain fixed.
- Calibration targets have separate fitting, checkpoint-selection and outer
  tuning roles. Outer observed-choice NLL freezes pipeline settings before fresh
  test scoring. Rare labels, oracle probabilities and test outcomes do not select
  them. Both checkpoint rules, all controls and null diagnostic support remain.
- Rare diagnostic KL difference is predeclared alongside overall NLL/KL, with
  per-seed signed values, mean/range, direction counts and supported trials/people.
  Matched rules still differ in objective and regularization. Teacher similarity
  is descriptive; no causal, significance or convergence inference is claimed.
- A fixed 4,000-step horizon and wider L2 grid are a combined robustness change,
  not a training-duration ablation. The one-calibration limitation, majority-led
  tuning, extra checkpoint-selection opportunities and separate 03/04 reporting
  are disclosed. No adaptive expansion based on test scores is authorized.
- Distinct output names, resume instructions, incomplete-result guards, soft
  timing limits and training-only T4 accounting are consistent with the engine.
- The secure downloader and running-kernel import checks are preserved. The
  readiness guard precedes runtime setup, full revision/source pins are required,
  and prior launchers remain unchanged.

Independent execution:

- Config plus baseline suite: **24 passed**.
- New/bootstrap/baseline focused suite: **59 passed**.
- Full suite: **198 passed, 3 CUDA-only skipped**, one expected duplicate-ZIP
  fixture warning.
- Independently ran the configured smoke and a completed resume in a temporary
  CPU folder. Recomputed group NLL, KL, supported diagnostic KL, trial support and
  primary paired differences from saved observations and probabilities; agreement
  within 1e-15. States were finite; resumed prediction arrays were bitwise equal.
- Inspected the actual five-cell executed notebook artifact from local validation:
  execution counts 1–5, no error outputs, correct Markdown newlines, signed focal
  values/counts and supported/unavailable seed presentation.

## Final release check

**PASS. No unresolved findings.** The independent reviewer checked:

- Exact runtime `bba8ce3e712173802f705650b27356fedbdea8fd` extracted from local Git.
  Package hash `63c789f6b9ef1cb7c8d32bd726ecfbfcf2c44aa3ac48ed5da18c5096cb008210`
  matches the final settings and launcher. Every numerical module, both new
  study configs, builder and verifier match the reviewed implementation.
- The readiness check precedes runtime setup. Only the pinned
  commit and readiness constants differ between the runtime's staged notebook
  and the shipped launcher. Notebooks 01–03 remain byte-identical to `31bd42b`.
- The final config and notebook suite passed **15 tests**.
- Final executed artifact
  `04_colab_robustness.final-pinned.executed.ipynb` matches all five
  shipped code cells, with execution counts 1–5 and no error outputs. Smoke,
  completed resume and the separate bounded software-check study completed.
- The exact archived runtime preflight passed **197 tests, with 4 skips**: three
  CUDA-only checks and one local-Git archive check that requires a checkout.
- Signed diagnostic differences, direction counts, missing-support handling,
  the fixed-control plot caption and all three displayed figures were inspected
  successfully. The inherited figure label is clarified without numerical changes.

The local verification ran on CPU. Colab runs its GPU preflight before the full study.
