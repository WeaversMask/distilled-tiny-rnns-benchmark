# Pilot methodology

This document defines the initial feasibility experiment. It does not preregister the complete research study. The dated proposal contains hypotheses; confirm and freeze a full analysis plan before running an inferential grid.

## Generating process

There are two first-stage actions, two second-stage states, and two actions per second-stage state. First-stage action a reaches state a with probability 0.7. Each second-stage action has an independent drifting reward probability, reflected into [0.25,0.75] with Gaussian increments of standard deviation 0.025.

Hybrid agents combine a model-free first-stage action value and a model-based value computed from the known transition matrix and the maximum learned second-stage action value. First- and second-stage choices use a softmax with inverse temperature beta. The learning rate alpha is shared by the TD updates, the eligibility trace is fixed at 0.5, and initial action values are 0.5. Alpha is drawn from Uniform(0.15,0.65), beta from Uniform(2,6), majority model-based weights from Uniform(0.5,0.9), and rare weights are zero. These choices are synthetic pilot assumptions, not calibrated Gillan parameters.

The simulator stores generating probabilities before sampling the current choice. The reference policy uses model-based weight 0.75 with the same alpha, beta, and observed history. It is useful for identifying policy disagreements but is not the true population-average policy.

## Prediction and masking

At trial t, the four input features are trial t−1's first action, reached state, second action, and reward, coded as −1/+1. The first input is all zero. Models output one logit for P(first-stage action = 1). The hidden state starts at zero for every person's sequence.

Each subject has independent seeded interspersed train/validation/test masks, approximately 60/20/20. The target for a held-out trial contributes no direct training loss. Its observed action can appear in later causal history inputs. Thus metrics measure conditional next-choice prediction on familiar participants. Full histories are forwarded through the network in time order, with losses masked; trajectories are not shuffled.

The teacher minimizes pooled observed-choice binary cross-entropy. The solo students minimize each person's observed-choice loss. Distilled students minimize alpha × T² × BCE(sigmoid(teacher_logit/T), student_logit/T) plus (1−alpha) × observed-choice BCE. The soft BCE differs from KL by a term independent of student parameters. Teacher targets are detached, and both terms are restricted to training targets. Here alpha is `teacher_prop`, separate from the agent's learning rate. Validation always uses observed-choice NLL. The teacher selects one best state; independent students each select their own best state. Solo and distilled models start from identical weights.

## Evaluation

All reported per-person NLL and KL metrics use only test positions. KL is D_KL(true generating policy || fitted policy) on observed histories. Diagnostic trials additionally require an absolute difference above 0.2 between true and reference choice probabilities. Zero diagnostic support is represented by null, never by a zero score. Group summaries weight subjects equally; diagnostic counts are reported separately.

The figure's x coordinate is solo NLL minus distilled NLL (positive means better prediction). Its y coordinate is distilled KL minus solo KL (positive means worse fidelity). Points are descriptive; no significance, prevalence trend, convergence, or mechanism claim follows from one short pilot. The plot can diagnose the measurement pipeline even if models are underfit.

## Reproducibility and limits

Fixed seeds control simulation, masks, and initialization. Checkpoints retain current training state and separately selected states, optimizer state, RNG states, and validation history. Interrupted training resumes the same validation schedule. Reusing a run folder with a changed configuration or source is rejected. Exact floating-point equality across GPU models, library versions, or CPU/GPU is not promised. Checkpoint/input hashes may reject a cross-device change that alters teacher predictions; use a new run folder in that case.

Each session preserves its complete package inventory under a content hash and records that path in the manifest. Earlier inventories remain available when packages change. Cumulative checkpoint time has no per-device breakdown, so the T4 cost proxy is emitted only if every recorded session used a T4. Mixed or unknown device histories explicitly receive no cost estimate.

The full experimental design must add mechanistic agents, stronger classical baselines, free rollouts and fixed probes, multiple seeds, convergence checks, and equivalence to the published pipeline before it can support the proposal's claims.
