# Independent review: first Colab feasibility pilot

Review date: 2026-10-29 (Europe/Berlin).

## Verdict

**Suitable for a guarded first Colab feasibility pilot, provided the notebook's verification tests and GPU smoke/resume check pass in that Colab runtime.** I found no implementation defect that blocks a fresh, single-runtime T4 pilot. This is not certification of CUDA execution, GPU timing, convergence, strategy recovery, or the full research proposal.

A fresh full local test pass could not be reproduced in the current filesystem state. The initial full-suite invocation stopped during collection with an installed-dependency file-read timeout. Subsequent diagnostics found macOS `dataless` dependency sources and bytecode caches, including notebook and symbolic-math dependencies. These are local environment limitations, not numerical test failures demonstrating a project defect. Fresh targeted checks and inspection of the preserved execution are described below; their evidence must not be conflated.

Implementation, configurations, notebook and bundle were not edited. No packages were installed, no large experiment or GPU run was launched, and nothing was committed or published. The preserved proposal was treated as reference material.

## Prioritized findings

### P2 — Cost attribution becomes incorrect when a run moves between device types

Locations: `src/distilled_rnns/training.py:188`, `src/distilled_rnns/training.py:234`, `src/distilled_rnns/training.py:273`; `src/distilled_rnns/experiment.py:181–191`.

Checkpoint elapsed time accumulates across invocations without device attribution. The report converts the entire accumulated duration using the **current** session's GPU name. A time-limited teacher trained partly on CPU and resumed on a T4 can therefore include CPU training time in the T4-unit proxy. A resume on another GPU also cannot retain an accurate attribution of prior T4 time. This conflicts with the report's statement that CPU time is not a GPU cost estimate.

This does not affect predictions and does not block a fresh T4-only pilot. Keep the initial cost interpretation to a run trained entirely on the same GPU type. Before supporting mixed-device cost comparisons, retain phase durations per session/device and convert only measured T4 portions, or return a null cost proxy for mixed-device histories. The supplied 1.07 units/T4-hour and 200-unit monthly allowance are assumptions supplied by the user, not independently measured billing facts. The allowance is not enforced by this code.

### P3 — Resume overwrites the previous complete package inventory

Locations: `src/distilled_rnns/experiment.py:123–129`; `src/distilled_rnns/experiment.py:51–60`.

Each invocation overwrites `environment-freeze.txt`. The manifest retains per-session Python/platform and the three principal package versions, but not the complete earlier package inventory. If Colab changes other packages between disconnects, the exact earlier environment is lost. Source/configuration checks do not reject such an environment change.

This is not a blocker for the first pilot. For longer experiments, save a separate package inventory for each session, or retain inventories under content hashes referenced by the manifest. Cross-library and cross-device bitwise reproducibility should remain explicitly unsupported unless separately established.

## Verification performed in this review

All Python checks used this project's `.venv/bin/python` (Python 3.13.0, PyTorch 2.15.0, NumPy 2.5.4). CUDA availability was false.

1. **Fresh simulator and metric tests:** `python -m pytest -q tests/test_simulation.py tests/test_evaluation.py -k 'not quadrant_plot_written'` completed with **42 passed, 1 deselected**. The plotting test was excluded from this focused run; this is not the complete suite.
2. **Fresh configuration/device and notebook tests:** the invalid-configuration parametrization, unknown-key rejection, and explicit-CUDA rejection tests completed with **12 passed**. Separately, `python -X pycache_prefix=distilled-review-pycache -m pytest -q tests/test_notebook.py` completed with **2 passed**, using a temporary bytecode-cache location to avoid unavailable existing cache files. Together with item 1, these are **56 freshly passing focused tests**, not a complete suite pass.
3. **Independent scalar simulator replay:** a separately written scalar calculation replayed 30 people × 50 trials. Both the generating policy and the fixed-weight reference matched the saved simulator probabilities, with maximum absolute error **4.44e-16**. This calculation did not call the simulator's probability/update helpers or its replay helper.
4. **Independent GRU and loss checks:** compared the batched students with separate ordinary PyTorch GRUs at hidden sizes 1, 2 and 5, using four subjects, 13 trials, unequal training-mask counts, teacher mixing 0.6 and temperature 2.3. Forward outputs matched at `rtol=1e-11, atol=1e-12`; all parameter and input gradients matched at `rtol=1e-10, atol=1e-12`. Teacher gradients stayed detached. Perturbing inputs from trial 7 onward left earlier outputs exactly unchanged. These comparisons used NumPy assertions, avoiding an unrelated optional dependency import made by `torch.testing.assert_close` in this environment.
5. **Bundle and notebook checks:** the bundle checksum matches its sidecar. Nineteen runtime/source files (source modules, tests, configs, scripts, notebook and package metadata) are byte-for-byte identical to the current project. Archive paths are relative and contain no parent traversal. A temporary extraction produces the same source fingerprint. The notebook has five clean Python code cells, current source pinning, and tests preceding smoke training.
6. **Preserved execution evidence:** independently parsed `artifacts/01_colab_pilot.executed.ipynb`. It has execution counts 1–5, no error outputs, the current source fingerprint, recorded **73 passed, 1 skipped**, and a smoke/resume success message. This corroborates the earlier execution record; it is **not a fresh end-to-end notebook execution by this reviewer**.
7. **Saved full-dimension shape run:** regenerated the saved 1,000-person × 100-trial dataset from its manifest and matched every simulator array exactly, including 20 rare subjects. All three `.pt` checkpoints load on CPU, have two completed steps, final-step validation, finite current weights, selected states, optimizer state and RNG records. This independently checks saved artifacts and dimensions, not adequate training or uninterrupted/resumed equivalence on a GPU.
8. **Full-suite attempt and limitation:** the ordinary `python -m pytest -q` failed collection with `TimeoutError: [Errno 60] Operation timed out` while importing `traitlets.utils.importstring` through `nbformat`. A repeat with pytest plugin autoload disabled failed similarly. A temporary review-only import diagnostic confirmed evicted dependency sources/caches (including `nbformat/v4/__init__.py`, `mpmath/ctx_mp_python.py`, and later SymPy modules). A diagnostic run excluding notebook tests, with a source-cache fallback and explicit rejection of unavailable dependency files, returned **15 failed, 56 passed, 1 skipped**; failures were dependency imports (Matplotlib and SymPy, including imports made by PyTorch comparisons/optimizers). This diagnostic result is not an ordinary full-suite run. No dependency installation or project workaround was made. Therefore I do **not** claim a fresh 73-pass suite, fresh Adam/restart equivalence, or fresh notebook execution from this review.

Reviewed source fingerprint:

`23fa24bde4f875f795bc4ef220b482e7f9752d724376ada0c0bf6ec16132c444`

Reviewed upload-bundle SHA256:

`d4f8f121a16a807db60af0bdf491721841b61389ef6ec4e57e61aa871c7a0b0c`

This review file was added after that bundle was built. The bundle's runtime/source files match the reviewed implementation; the bundle does not yet contain this new review document.

## Audit conclusions by area

| Area | Conclusion and evidence |
| --- | --- |
| Simulator equations and ground truth | The 70/30 transition matrix, model-based state maxima, hybrid weighting, softmax sign, old-value TD errors and fixed 0.5 eligibility trace agree with the stated synthetic process. Probabilities are recorded before sampling the current action; reward drift is reflected. Scalar replay and saved-data regeneration passed. See `simulation.py:66–88`, `simulation.py:135–171`. Storing reward-probability paths is unnecessary for replaying the first-stage policy conditional on saved outcomes; the implementation does not claim to export those environmental paths. |
| Causality and split interpretation | Inputs are exactly the preceding trial's four observations, with a zero initial input (`simulation.py:157–159`). Masks independently partition each person's targets (`evaluation.py:70–88`). Full histories remain available, so a held-out action can affect later inputs and later training losses. This is an interspersed conditional-prediction estimand on familiar participants, not an untouched session, forward-only forecasting assessment, or unseen-participant test. The README/methodology disclose that distinction. |
| Independent students and gradients | Separate parameter slices and zero initial hidden states implement independent GRUs (`models.py:42–73`). The reset gate acts after the hidden affine transform, matching PyTorch. Fresh independent forward/gradient checks passed. Summing per-subject mean losses preserves each independent student's gradient scale (`training.py:52–59`); no cross-person gradient clipping couples them. The stock Adam-equivalence test exists at `tests/test_models_training.py:85–99`, but could not be freshly completed under the dependency limitation. |
| Masking and distillation | Hard and soft losses are both masked; teacher logits detach; the soft term has the stated temperature-squared factor (`training.py:44–59`). Validation always uses observed-choice BCE. Solo and distilled initializations share a seed (`experiment.py:140–157`). The teacher contributes training-position targets; test metrics do not select weights. Fresh mixed-loss gradient checks passed. |
| Validation selection | Teacher selects one pooled state; students select per-subject parameter slices and steps (`training.py:201–221`). Step zero and the requested final step are evaluated. Latest optimization state is saved separately from selected prediction weights. The focused two-subject selection test at `tests/test_models_training.py:137–151` is appropriate; preserved execution records the earlier suite passing. |
| Restart and CUDA portability | Checkpoints include model/optimizer, selected state, schedule/history and RNG. Signatures cover tensors and relevant fit settings; run manifests additionally bind the full config and source. CPU RNG restoration and Adam scalar-step restoration explicitly move tensors back to CPU after device mapping (`training.py:172–192`). Time-limited stops preserve the validation schedule. The CUDA-specific test exists at `tests/test_models_training.py:235–250` and will run in Colab. No CUDA behavior was measured here; cross-device teacher-target byte changes may correctly reject a downstream checkpoint, as documented. |
| Budget limits | The configured 600 seconds is a per-phase, per-invocation training-loop cap, checked between steps (`training.py:198–251`). It can be exceeded by one step, evaluation or checkpoint I/O, and does not cap notebook setup, idle billing, or the monthly account total. A capped phase returns incomplete and prevents later arms/report interpretation (`experiment.py:159–167`). Repeating the invocation supplies another time allowance. Documentation accurately describes these practical bounds. |
| Colab bootstrap and Drive paths | The README upload location matches `/content/drive/MyDrive/distilled-tiny-rnns/colab_project.zip`; results use its `runs/` folder. Bundle extraction rejects traversal/symlink entries; source is pinned before installation. Tests, explicit GPU detection, a small GPU run, and completed-run reload precede the pilot (`scripts/build_colab.py:30–114`). Those routes are correctly ordered on inspection. Actual Drive mounting, current-image installation, FUSE checkpoint behavior and CUDA execution remain untested here. Use one runtime per Drive output folder; the local lock cannot coordinate different runtimes. |
| Metrics and scientific claims | Test-only Bernoulli NLL and forward KL, subject-weighted group means, explicit null diagnostic support, fixed 0.75 reference, and quadrant signs are internally consistent (`evaluation.py:37–67`, `evaluation.py:141–194`). Fresh metric tests passed. The main figure uses all test-trial KL; the diagnostic subset adds NLL only. Short-run prediction/fidelity differences are descriptive, not significance, mechanism or population-prevalence evidence. |

## Conditions for the first pilot and deliberately deferred work

Start with the reviewed notebook and bundle in a fresh Colab T4 runtime. Require its full tests (including the CUDA-only test) and smoke/resume assertion to pass before accepting the pilot's output. Save the resulting manifest, package inventory, checkpoints, predictions and report. Interpret a time-limit return as incomplete, and retain the same settings/run folder for resume. Actual T4 runtime, memory and account-unit consumption must be read from that run; no GPU-hour guarantee follows from this review.

The current implementation does not need a mechanistic rare agent, released tinyRNN equivalence, classical/hierarchical baselines, second-stage prediction, fixed probe histories, free rollouts, pull indices, dynamical regressions, multiple-seed inference or the full grid merely to test whether this first synthetic pipeline runs. Those remain prerequisites for the corresponding scientific claims in the later study, as the README, methodology and roadmap already state. No human-data arm or external publication is authorized by this review.
