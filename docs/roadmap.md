# Study milestones

1. **Feasibility.** The [initial pilot](methodology.md) checked the simulator, batched GRU and checkpoint workflow. The [follow-up](followup.md), [robustness study](robustness-results.md), [oracle control](oracle-control-results.md) and [training-budget check](budget-check-results.md) examined sensitivity to selected training choices.
2. **Method alignment.** A [small comparison fixture](published-method-alignment.md) evaluated the released tinyRNN implementation and the project’s first-stage model on the same simulated histories.
3. **Prospective study design.** The project implemented the [reward-cue agent](rare-mechanism.md), verified [classical controls](classical-baselines.md), and froze the [version 1 protocol](main-study-protocol.md) before training the first tranche.
4. **Completed 100-trial sweep.** Notebook 07 ran the [2% baseline](main-study-first-tranche-results.md). Notebook 08 extended it to [5%, 10%, and 25% rarity](rarity-tranche.md). These 24 populations are the scope of the [current results](../review/notebook08-audit/report.md).
5. **Remaining design cells.** The frozen grid includes 72 populations at 200, 350, and 1,000 trials that have yet to be run. A longer-history study would require its own device timing and result verification.
