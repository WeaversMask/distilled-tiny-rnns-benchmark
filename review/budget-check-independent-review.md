# Independent bounded-budget review — 2026-11-03

Internal reviewer: GPT-6.1 Sol.

## Implementation verdict

**Final PASS: no open findings.** Both the implementation and the exact pinned
release execution passed independent review.

The reviewer independently ran all **86** budget tests (44 runtime and 42
notebook tests), all passing. It also ran the strengthened parent validator
against the actual downloaded `oracle-control-v1`: all **77 files** and all
five original populations passed. Existing tracked numerical modules and
notebooks 01–05 were verified byte-identical to release
`393a8ec03791cb6a2f32cbd63b2f0638d667935a`.

## Scope reviewed

- Exact production and smoke parent settings, original-source identity,
  generated data and split masks, package inventories, finite checkpoints,
  Adam configuration/state, current and selected predictions, stored reports
  and aggregate reproduction.
- Continuation of current weights, optimizer/RNG, cumulative history and both
  selected states; fixed teacher and solo references; immutable parent
  snapshots; interruption and completed-arm resume; no extra training after
  completion; child-only incremental compute attribution.
- Paired 4,000-versus-8,000 reporting, explicit contrast signs, both selection
  rules, equal-person/equal-seed summaries, unavailable support, per-seed
  directions, fixed references and the prohibition on pooling horizons as new
  independent populations.
- Secure pinned bootstrap, stale-import and GPU checks, visible preflight
  errors, genuine smoke-parent creation, repeated third-cell execution in live
  and fresh namespaces, fixed production routing, and the five-cell verifier.

The one requested documentation correction was resolved: the oracle-results
report's follow-on recommendation now links to the agreed foundation plan
(published-method comparison, stronger baselines, distinct rare mechanism and
protocol freeze), rather than directing the project immediately into teacher
interventions.

## Exact final release

The reviewer independently hashed and inspected runtime archive
`baaf93518fae4eec86b3580b74aa2bbcdac0698b`. Its full-package source fingerprint
is `243a4e1c70f78dbe8d31ec0ebe48a65f6eeba60c54719c597b40081e44a48822`.
The runtime, production and smoke configs, and bootstrap match the reviewed files.
The historical launcher pins this commit and hash, checks readiness before
setup, and has empty execution outputs. Its builder reproduces it byte-for-byte.

All shipped notebook cells, including Markdown, exactly match the executed
artifact `06_colab_budget_check.final-pinned.executed.ipynb`.
All five code cells completed without errors. The actual archived preflight
reported **354 passed, 6 skipped**: three CUDA checks and three Git-checkout-only
archive checks. The working checkout had **357 passed, 3 CUDA-only skipped**.
The real 4→8 continuation, immutable parent, completed resume and final report
were independently inspected and passed.

Shipped notebook SHA256:
`1f8b51704ad5a338e57e3826064f5f955875053cd5eb8c1dff703ee252d4cc97`.
Executed artifact SHA256:
`767e7d99fd346b5876468e4872fb8b63b4c6bd1dfa09176ff9305691dc861fbf`.

This review assesses the software and bounded exploratory design. It does not
establish convergence, a capacity explanation, statistical significance or a
research conclusion. Local execution uses CPU; CUDA checks and actual T4
behavior remain for Colab before production training.
