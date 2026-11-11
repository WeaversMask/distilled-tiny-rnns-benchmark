# Notebook 08 local verification

Verified 2026-11-05 on the fixed grid adapter source
`a4582314a2b34289883f526366298f6604bb4c1e327856d068ee91e2c494c9df`.
The original main-study source retains
`b5c5aa888e9b9ba00e9e8c69486c740a40efaf134bcda41397d8980fc4944372`,
and the prospective protocol retains
`1e8762df6fc01919dd3cfc432e96b78ed1d3fe1a25c26859edd9674cd501821e`.

The working checkout executed all five actual notebook cells in a fresh local
CPU kernel. Preflight passed 446 tests with six CUDA-only skips. The original tiny
baseline, all three tiny rarity cells, both strategies, completed resume and
final tables completed. The local production cell uses only development smoke
seeds and budgets. The 18 main-study populations remain unevaluated.

The runtime's 16 targeted tests passed with one CUDA-only skip. The separate
independent reviewer passed 32 tests with one CUDA-only skip. Checks cover
strict frozen seed/recipe selection, percell summaries, incomplete-state
reporting, interruption/resume equality of optimizer/RNG/current/selected
states, corruption and alias rejection, canonical locks, actual atomic-write
failure recovery and completely immutable completed reruns. Review also
injected final summary/manifest failures and preserved all 48 already-saved
checkpoint files while recovering.

The production baseline validator matched every one of the 87 previously audited
files in the actual downloaded first tranche without changing their bytes.
Portable references preserve exact summary/protocol bytes and bind the full
original inventory. Dynamic smoke validation is confined to the exact old
smoke configuration. All 20 old numerical Python files, frozen protocol and
notebooks 01–07 remain byte-identical to the starting revision.

The adapter fixes a narrow initialization recovery edge for actual `.json.tmp`
staging paths around the unchanged legacy engine, including the original tiny
baseline smoke. This affects retry handling and preserves fitting behavior.
It rejects arbitrary or mismatched staging contents.

The next tranche is the predeclared 5%, 10%, 25% rarity slice at 100 trials. Timing
reference comes from the measured first tranche and is labeled as a scheduling
estimate. Baseline compute is separate. The final tables retain diagnostic
scores and all models alongside the primary assay comparison. Differences
across rarity levels use distinct populations and receive descriptive summaries.

[Independent review](rarity-independent-review.md) ·
[Read-only baseline verification](rarity-baseline-local-verification.json).

The exact runtime archive at commit
`87f5068a637377089fe0ea7087a13211a9ea8d2b` executed all five final shipped
cells. Preflight passed 443 tests with nine expected skips (six CUDA-only and
three Git-metadata checks absent from an archive). Shipped and executed cell
sources match exactly, shipped outputs are clean, and the final independent
review passed with no open findings. The archived runtime and notebook release
each matched all 172 tracked files byte for byte. The local execution used CPU.
CUDA and Drive checks run in Colab. [Archive record](rarity-archive-verification.json)
