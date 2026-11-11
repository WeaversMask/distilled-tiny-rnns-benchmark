# Original research proposal and related work

October 28, 2026

This proposal records the motivation before the feasibility pilot. The [frozen main-study protocol](main-study-protocol.md) defines the implemented study, and the [100-trial results](../review/notebook08-audit/report.md) document the completed comparisons.

## Research question

Ji-An, Benna and Mattar fit a pooled GRU to choices and train small per-person GRUs to match its predicted choice probabilities. Pooling can improve prediction for people with short histories. It could also make a teacher’s predictions less faithful to a rare strategy, and a distilled student could inherit that change.

Simulated populations provide a way to measure the two outcomes separately. Each person's generating policy gives a reference for fidelity, and their sampled choices give held-out prediction scores. The proposal was to test whether the gains and losses varied with strategy prevalence or history length.

## Related work

The table records the studies considered during the original proposal. It describes how they motivated the simulation question.

| Work | What it did | What it leaves open |
| --- | --- | --- |
| [Ji-An, Benna & Mattar, Nature, Jul 2025](https://www.nature.com/articles/s41586-025-09142-4) | Trains a teacher on all subjects, then a tiny GRU per subject to match the teacher's choice probabilities. On human data the students used 5 to 20 units and beat classical models with few trials | Whether a student keeps a rare person's strategy |
| [Commentary on Ji-An et al., bioRxiv, Oct 2025](https://www.biorxiv.org/content/10.1101/2025.10.30.685524.full.pdf) | Argues RNNs track individuals through in-context adaptation, so student dynamics "may not purely reflect cognitive processes". Simulates a model-free agent that switches between engaged and disengaged | No distillation in the simulation; says the student question "warrants further examination" |
| [Katahira, "Excessive flexibility?", Comput. Brain Behav., Jul 2025](https://link.springer.com/article/10.1007/s42113-025-00254-8) | RNNs trained on pooled data adapt to each person within a session, and individually fitted models sometimes match or beat them | Trains on observed choices; no teacher or student |
| [Song, Niv & Cai, CogSci 2021](https://escholarship.org/uc/item/3wj7w4x7) | An RNN with a participant embedding beats the best cognitive model on a reward-learning task | No distillation |
| [Dezfouli et al., NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/file/e077e1a544eec4f0307cf5c3c721d944-Paper.pdf) and [Miller et al., NeurIPS 2023](https://proceedings.nips.cc/paper_files/paper/2023/hash/c194ced51c857ec2c1928b02250e0ac8-Abstract-Conference.html) | An encoder maps each person to a latent vector that sets a GRU's weights (2019); an RNN penalized for carrying extra information recovers known cognitive models from synthetic data (2023) | Neither distills a pooled teacher into per-person students; no rare-strategy test |
| [Lukasik et al., "Teacher's pet", arXiv 2021](https://arxiv.org/abs/2106.10494) | Distillation can raise average accuracy while hurting subgroups with few samples, because the student inherits and amplifies the teacher's errors | Image classification; no behavior models and no known ground-truth strategies |
| [Mohammadshahi & Ioannou, arXiv 2024](https://arxiv.org/abs/2410.08407) | On image data, up to 41% of classes change significantly under distillation; higher distillation temperatures improved fairness measures | Classification only. Shows temperature is a lever, which hypothesis 5 uses |
| [Wang, "Knowledge distillation must account for what it loses", arXiv May 2026](https://arxiv.org/html/2604.25110v2) | Position paper: treats distillation as a lossy projection and lists subgroup and long-tail losses | No experiments; no behavior models |
| [Ramesh et al., arXiv 2025](https://arxiv.org/abs/2505.15442) and [Hu et al., CogSci 2026](https://arxiv.org/abs/2601.05019v1) | In language models, distillation raises accuracy without preserving the teacher's reasoning process, and distilled reasoning students lose human-like difficulty scaling | Language models; fidelity to the teacher, not to a known generating strategy |

## Initial hypotheses

| # | Hypothesis | Prediction | Settled by | Counts against it |
| --- | --- | --- | --- | --- |
| 1 | **Pull** | A rare-strategy person's student drifts toward the majority, more at low prevalence and with few trials | Pull index by prevalence and trial count | Pull index near 0 even at 2% prevalence and 100 trials |
| 2 | **Silent distortion** | In some regimes the student beats a solo tiny RNN on held-out likelihood while its strategy fidelity is worse | Likelihood gain against fidelity loss for each rare-strategy person (the quadrant plot) | No one lands in the "better prediction, worse strategy" quadrant beyond noise |
| 3 | **Masking** | The error sits on the few trials where the strategies disagree, so overall likelihood hides it | Likelihood on trials where the true policies differ by more than 0.2 in choice probability, against all trials | The gap on those trials is no larger than overall |
| 4 | **Source** | The pull starts in the teacher, not in the student step | Fidelity of the teacher's own per-person predictions, against the student's | The teacher is faithful and only the student is not |
| 5 | **Fixes** | Mixing the person's own choices into the student's targets (the authors' teacher\_prop setting) cuts the pull at a small cost in likelihood | Sweep teacher\_prop (0, 0.2, 0.8, 1) and distill\_temp (1, 2, 4, 8) | No setting cuts the pull without giving up most of the likelihood gain |

The proposed capacity and distillation-setting sweeps are separate from the completed 100-trial study. The final protocol also formalizes the behavioral assay, diagnostic support, selection rules, and controls.

## Planned grid

| Factor | Levels | Why |
| --- | --- | --- |
| Rare-group share | 2%, 5%, 10%, 25% | Pull should grow as the group gets rarer |
| Trials per person | 100, 200, 350, 1,000 | Distillation helps most with few trials |
| Rare-group kind | Parametric, mechanistic | Separates shrinkage from a wrong mechanism |
| Seeds | 3 per cell | Variation across simulated populations |

The four prevalence levels, four history lengths, two rare strategies, and three population seeds produce 96 planned populations. The completed study covers the 24 populations at 100 trials. See the [current milestones](roadmap.md) and the [comparison with the published implementation](published-method-alignment.md) for the scope of the implementation.
