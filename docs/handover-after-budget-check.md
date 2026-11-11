# Notebook 06 development record

The [Notebook 05 results](oracle-control-results.md), [Notebook 06 protocol](budget-check.md), and [Notebook 06 results](budget-check-results.md) contain the scientific comparisons. The [foundation plan](foundation-plan.md) records which checks preceded the [frozen main-study protocol](main-study-protocol.md).

Notebook 06 continued the two soft-target arms from 4,000 to 8,000 updates on the five existing populations. It kept the teacher, solo fits, data, and targets fixed. The cumulative checkpoint selection used the entire 0–8,000 update window. This changes both the training budget and the number of available checkpoint candidates.

## Historical release and correction

- Original runtime commit `baaf93518fae4eec86b3580b74aa2bbcdac0698b`. Original launcher commit `f5ec95d35302dd73b79da89d1d054ea1f17f0860`.
- Corrected runtime commit `a97cb3f2e92fbce06c240484b3a52d012f56a7b9`. Corrected launcher commit `dc5ec43`.
- Corrected runtime source SHA-256 `a7394cdff009e97f1130e135e8439e5ae70295f0fb7b29e01eba771831a659a0`.
- Audited parent source SHA-256 `c68c9b2918c7199f0396a9a3bc28f98f1bd28f59469adf6524bfe6b67ac6c3d9`.

The original production cell stopped before extension training when CPU validation of a T4-generated parent prediction exceeded the existing tolerance. The same predictions reproduced on T4. The corrected launcher checks forward predictions on the selected run device while retaining the CPU checkpoint structure checks and original tolerances. [Device correction and verification](../review/budget-colab-device-fix.md)

The [release verification](../review/budget-check-local-verification.md) and [internal implementation review](../review/budget-check-independent-review.md) document the CPU smoke, resume, and parent-preservation checks. The correction produced a new smoke run name, `budget-check-smoke-a7394cdff009`. Production kept `budget-check-v1` and the `oracle-control-v1` parent.
