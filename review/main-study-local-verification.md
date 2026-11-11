# Notebook 07 local verification

Verified 2026-11-04 using Python 3.13 and PyTorch 2.15.0 on CPU.
The main runtime source fingerprint is
`b5c5aa888e9b9ba00e9e8c69486c740a40efaf134bcda41397d8980fc4944372`.
The prospective protocol retains SHA256
`1e8762df6fc01919dd3cfc432e96b78ed1d3fe1a25c26859edd9674cd501821e`.

The working checkout executed all five actual notebook cells in a fresh kernel.
Its preflight completed 414 passing tests and five CUDA-only skips. Both smoke
rare kinds, every model, the three-choice assay, completed-run immutability and
the separate local study output completed. The notebook's primary tables show
both coordinates, both checkpoint rules, every seed, all model scores,
diagnostic support and timing. The fixture uses smoke seeds and budgets.

Twenty runner checks passed with one CUDA-only skip. Interrupted training
matched uninterrupted predictions, current/selected weights, Adam state,
learning history and RNG exactly. Completed reruns add no compute and preserve
all file bytes. Source/configuration changes, corrupted saved artifacts,
invalid optimizer settings, concurrent writers and source-directory aliases
are rejected. Initialization and report-finalization crash windows recover
without discarding checkpoint trajectories. Unrecorded wall time from abrupt
sessions remains explicitly unavailable with coverage counts.

Scientific checks independently reconstructed true policies for both rare
kinds, training-only baselines, fixed posterior policies in the assay,
per-person metrics, group weighting, primary signs and diagnostic nulls. The
release review is recorded separately. [Independent review](main-study-independent-review.md).

The existing numerical package retains fingerprint
`a7394cdff009e97f1130e135e8439e5ae70295f0fb7b29e01eba771831a659a0`.
All its files and notebooks 01–06 remain byte-identical to the previous release.
The original downloaded scientific results were read-only throughout this work.

The exact runtime archive at commit
`29eb07627f0d8ea20e5b5964e7c44287d464f857` executed all five final shipped
cells successfully. Its preflight passed 411 tests with eight expected skips
(five CUDA-only and three Git-metadata checks absent from an archive). The
shipped and executed cell sources match exactly and shipped outputs are clean.
The final internal release review passed with no open findings. The archived
runtime and notebook release each matched all 144 tracked files byte for byte.
The local execution used CPU and provides software verification. CUDA and Drive
checks run in Colab. [Archive record](main-study-archive-verification.json)
