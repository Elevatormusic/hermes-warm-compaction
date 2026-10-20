# Compaction benchmark: pre-registered plan and decision rule

Status: written and committed before the first full run. The commit that adds this file comes before any
result of a full run. A later change to this plan must be a new commit that says why.

The code in `run.py` (`decide()`) applies this rule. The report prints each check as pass or fail.

## Question

Which compaction method keeps the facts that a coding agent needs, at what success rate, size, and time? The
methods are the WarmCompaction warm path, the fallback summary in the quoted and the relaxed format, each with
and without the work ledger, and the Hermes built-in compressor. The control is the full history with no
compaction.

## Workload

- Six synthetic coding sessions, about 23,000 to 100,000 estimated tokens (`fixtures.py`, `scenarios.py`). Each
  has 46 questions: goal 5, user instructions 5, command outcomes 9 (6 exit codes, 3 error lines), files changed
  5, identifiers 5, decisions 5, task state 5, next step 5, and 2 recent facts near the end of the session.
- Totals: 276 questions. Each scored category except "recent" has 30 or more questions (command outcomes: 54).
- The questioned facts (not the recent facts) are in the first part of each session, in front of the last 14,000
  estimated tokens. A verbatim tail of 10,000 tokens therefore does not hold them.

## Procedure

1. Each job is one fresh process: a clean Hermes checkout, a new temporary Hermes home, the plugin installed by a
   file copy, and the real Hermes agent loop (`AIAgent.run_conversation`) with no tools.
2. The agent sends one main request with the session as its history. Then the Hermes manual compress core
   compacts the history (`agent/conversation_compression_manual.py::compress_now`, which every Hermes surface
   uses for `/compress`). Change after registration (2026-10-20, PR review): the runs made with the harness
   commits `c195a40`, `fecb892`, and `f6dc770` called `tui_gateway.server._compress_session_history`, the private
   TUI gateway wrapper of the same `compress_now` core. Their reports state this entry and the Hermes commit.
3. A separate request asks the questions on the compacted history, at temperature 0, in the batches of the
   plan in the gold file (at most 12 questions in a batch; five batches of 9 or 10 for 46 questions). No gold rule
   passes on the prompt of its batch: the full prompt, a line, or a word of the instruction or of another question.
   A reply that is not a full answer (no text, a finish reason other than `stop`, or a missing or empty answer)
   fails the job; it is not scored as wrong. When a compaction does not compact (an abort or no change), no
   questions are asked.
4. Model: `qwen3.5-4b-mtp` in LM Studio (one parallel slot), for the main request, the compaction, and the
   answers. Context window setting: 131,072. Compaction threshold: 0.95, so no automatic compaction starts
   before the manual one.
5. Five replicates of each method on each fixture. Repeated compaction (drift): fixtures `s2-tidepool` and
   `s4-orchard`, methods `stock_hermes`, `warm_ledger`, and `fallback_relaxed_ledger`, three cycles, five
   replicates. A new chunk of work (half the fixture size) comes before cycles 2 and 3.

## Measures

- Success: the share of runs with a model-written summary (`warm`, `fallback`, or the built-in `model` path).
  A ledger-only summary, a built-in static fallback, an abort, and no change are not a success. Failure codes
  are counted.
- Accuracy by category: the share of correct answers, over the runs that compacted. Rules are in `scoring.py`.
  No model judges an answer. The report also gives accuracy with each run that did not compact scored as wrong.
- Retention (secondary, approximate): whether the fact text is still in the compacted history.
- Size: estimated tokens before and after (JSON characters / 4), and the server prompt count of the first
  question request.
- Time: wall time of the compress call (all runs) and of the main request, with per-run values.
- Intervals: 95% Wilson intervals for rates (pooled over questions; these do not include the variation between
  replicates), 95% t intervals for means. Paired comparisons use the exact McNemar test on the same fixture,
  replicate, and question.

## Decision rule

Checks (each pass or fail):

- H1 (ledger helps the warm path): `warm_ledger` minus `warm_no_ledger` is at least 10 points with McNemar
  p < 0.05, for both command outcomes and files changed.
- H1b (ledger helps the fallback): the same check for `fallback_relaxed_ledger` against
  `fallback_relaxed_no_ledger`.
- H2 (no regression): for goal, decision, next step, and task state, `warm_ledger` is not more than 5 points
  below `warm_no_ledger`, and not significantly lower (McNemar p < 0.05).
- H3a (match the built-in compressor where the ledger applies): for command outcomes and files changed, the upper
  bound of the `warm_ledger` interval is at or above the `stock_hermes` rate.
- H3b (no loss where the model summary applies): the same check for goal, decision, and next step.
- H4 (size): the mean estimated tokens after compaction of `warm_ledger` is at or below that of `stock_hermes`.
- S1 (warm success): the `warm_ledger` success rate is at least 90%.
- R1 (fallback format): the lower bound of the `fallback_relaxed_ledger` success interval is above the upper
  bound of the `fallback_quoted_ledger` success interval.
- D1 (drift): the `warm_ledger` accuracy at cycle 3 is not more than 10 points below cycle 1.

Verdicts:

- H1 and H2 pass: keep the ledger on by default. Else: the ledger does not meet the rule for a default.
- H3a, H3b, and H4 pass: warm path + ledger is competitive with the built-in compressor on this workload.
- R1 pass: the relaxed fallback format succeeds more often than the quoted format.

## Limits stated in advance

- Synthetic sessions from one generator. They do not prove behavior on real sessions.
- One small local model writes the summaries and the answers. A larger model can change all numbers.
- A rule-based score can miss a correct paraphrase. The full-history control shows the ceiling of the rules
  and the answer model.
- LM Studio does not report cached tokens. Cache reuse is unknown.
- The pooled Wilson intervals are too narrow when replicates of the same fixture are correlated.
