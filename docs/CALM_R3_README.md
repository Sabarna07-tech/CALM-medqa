# CALM-R3 — Calibration-Aware Reflective Routing & Reasoning

**What this is:** a research-grade implementation of your proposal, plus an honest
validation harness. It teaches the model *when not to reason*: a calibrated
fast pass answers confident questions, and only uncertain ones are routed to a
reasoning path. Run on top of the trained 7B student in `NOTE-RL_MedQA_v3_qwen7b.ipynb`.

## Run order

1. Finish the normal run so `student` is the validation-selected checkpoint.
2. Paste `calm_r3.py` as new cells at the end.
3. **Phase 0 runs automatically** (no training, ~20–40 min). Read its `VERDICT`.
4. Only if the verdict is **GO**, set `RUN_PHASE1 = True` and re-run the Phase 1 cell.
5. Artifacts are written to `paper_artifacts_v3_qwen7b/calm_r3/` and zipped to
   `calm_r3_artifacts.zip`.

## The one question Phase 0 answers (the validation)

The whole architecture pays off only if **the reasoning path is more accurate than
the fast path on the low-confidence subset**. Your earlier self-consistency result
(reasoning 0.493 vs single-pass 0.709) is a warning that this student's CoT may be
weak. Phase 0 measures it directly:

- buckets the test set into confidence deciles and prints `fast_acc` vs `reason_acc`
  per bucket (`phase0_buckets.csv`);
- sweeps the routing threshold τ and prints the accuracy-vs-compute frontier
  (`phase0_sweep.csv`, `phase0_acc_vs_compute.png`);
- prints **GO** if the best routed accuracy beats fast-only by >0.5 pt *and* reasoning
  beats fast on the bottom-30% confidence items; otherwise **NO-GO**.

A NO-GO is a real, publishable finding too ("for a one-shot-tuned student, reasoning
does not recover hard cases"), and it saves you from burning 12h on Phase 1.

## What I corrected from the original proposal

**1. Reflective Correction Loss.** The original `KL(P_fast || P_reason)` applied "only
when fast is wrong" makes the two paths *agree*, which drags reasoning toward the wrong
fast answer (neither side anchored to the label). The well-posed objective used here is

```
L = CE(P_reflect, y)                         on HARD items (fast wrong / unsure)
  + mu * KL( stopgrad softmax(P_fast) || softmax(P_reflect) )   on EASY items
```

i.e. supervise reasoning toward the gold answer where fast fails, and keep it
consistent with the confident fast prediction where fast already works.

**2. In-loop generation is too slow** for 2×T4/12h, so reasoning is conditioned on
**offline rationales** (generated once, greedy) rather than sampled every step.

**3. ACE = accuracy / tokens is degenerate** (the fast path generates ~0 tokens, so the
ratio explodes). Replaced by an **accuracy-vs-compute frontier** plus the honest
headline "matched or better accuracy at X% of the always-reason decoding tokens."

## Two-adapter routing (Phase 1)

- **Fast adapter** = your existing trained adapter (`default`).
- **Reflect adapter** = a second LoRA adapter trained with the corrected loss on the
  hard subset, conditioned on rationales. LoRA adapters are tiny, so both fit one T4.
- At inference, the calibrated fast confidence routes each question: `conf >= τ` → fast
  letter read; `conf < τ` → switch to the reflect adapter and reason. `phase1_routed.json`
  reports routed accuracy, fraction reasoned, tokens/question, and Δ vs fast-only.

## Budget knobs (top of `calm_r3.py`)

`REASON_TEST_N` (default 600; set 1273 for the full split), `REASON_MAXTOK`,
`HARD_N`/`EASY_N`, `ROUTE_GRID`, `RUN_PHASE1`. Every long loop has a `have_time()` guard.

## What to send me back

Paste the Phase 0 `VERDICT` block and `phase0_results.json` (and, if you run it,
`phase1_routed.json`). If GO, I will wire the real numbers and the accuracy-vs-compute
figure into the paper using `calm_r3_paper_block.tex`.
