# Agreed foundation plan — 2026-11-02

This records the user's agreed next steps before the main research study.
Maintain this plan across chats and notebooks; do not replace it with an
open-ended sequence of pilot refinements.

## Current position

Updated 2026-11-05. The first main-study tranche is complete and verified.
Its primary population means show better prediction and lower assay error
from distillation, with mixed secondary diagnostic results. The family-mixture
baseline captures cue behavior much better. [Detailed findings](main-study-first-tranche-results.md).

The bounded budget audit is closed. Representative published
code, classical/hierarchical fits and reward-cue mechanism checks now run locally
with independent verification. The prospective main-study design is recorded
before evaluating its populations.

| Agreed foundation item | Current evidence |
| --- | --- |
| Bounded 4,000→8,000 sensitivity | Both downloaded runs complete and verified read-only. Main oracle advantage persists. [Results](budget-check-results.md). |
| Published-method comparison | Pinned actual authors' code, native and project fits on shared synthetic histories, encoding and selection differences documented. [Comparison](published-method-alignment.md). |
| Stronger baselines | Individual and pooled generating-family grid fits, empirical-Bayes hierarchy, identifiable long-history recovery and short-history limits checked. [Baselines](classical-baselines.md). |
| Different rare strategy | Reward-cue simulator, probabilities, hand-derived diagnostic and hierarchical family mixture verified. [Mechanism](rare-mechanism.md). |
| Freeze main study | Version 1 defines comparisons, behavioral assay, seeds, recipe and stopping before new populations. [Protocol](main-study-protocol.md). |

Notebook 07 completed all six populations and its separate software smoke.
Notebook 08 schedules the next 18 populations: 5%, 10% and 25% rarity at
100 trials for both strategies. This preserves the declared recipe and all
controls. [Prospective tranche and run instructions](rarity-tranche.md). The
100-trial runner measured 118.86 minutes for six populations. Other history
lengths need their own runtime measurements for compute scheduling. The foundation evidence is bounded to synthetic checks and
the documented first-stage adaptation of the published method.

## Agreed sequence and completion criteria

1. **One bounded training-budget check.** Notebook 06 extends the
   two soft-target student arms from 4,000 to 8,000 steps across the same five
   populations. Continue copied current weights, optimizer/RNG and cumulative
   selected states. Preserve the original results. Stop at 8,000 regardless of
   outcome; no automatic escalation to another horizon or test-driven choice
   of the preferred horizon. This checks sensitivity to more optimization and
   more checkpoint candidates, not proof of convergence. Review the result and
   freeze the subsequent training procedure before fresh main-study data.
2. **Published-method comparison.** Run a small representative comparison with
   the authors' implementation. Verify input encoding, split roles, fitting,
   selection and relevant model size. Record deliberate differences and limit
   claims accordingly. Existing agreement with standard PyTorch does not
   establish agreement with the published research pipeline.
3. **Stronger scientific baselines.** Fit classical behavioural models,
   including the generating family, and a hierarchical model that shares
   information across people. Verify recovery in identifiable synthetic cases
   with sufficient information; document limits with short histories. Add a
   mixture comparison when the mechanistic population requires it. This
   distinguishes distillation-specific effects from sparse-data estimation
   and ordinary pooling/shrinkage effects.
4. **One genuinely different rare strategy.** Implement and independently
   verify a mechanistic rare agent and its true probabilities. Check its causal
   histories and diagnostic cases before interpreting training outcomes.
5. **Freeze the main study protocol.** Predeclare primary comparisons,
   baselines, prediction and behavioural-probe measurements, seed plan and
   stopping rules. Define what preserving a strategy means beyond conditional
   first-stage prediction. Existing pilot protocols remain unchanged.

Published-method and baseline preparation may overlap after the bounded check.
Items are complete when the representative scientific checks work and their
limits are documented, not when every conceivable control has been exhausted.

## Transition to the main research

After those pieces work in a small representative experiment, vary rarity and
history length, test generalization, and evaluate practical remedies. The large
grid, broad capacity sweeps, every teacher modification and optional human-data
arm are not prerequisites for beginning the main study. No human-data request
or message to another person is authorized by this plan.

Independent GPT-6.1 Sol review remains part of notebook release verification.
Keep original results and negative or mixed findings visible.
