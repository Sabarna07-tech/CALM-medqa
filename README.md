# CALM: Calibrated, Order-Invariant, Label-Efficient Medical QA on a Two-GPU Budget

Small-language-model medical question answering (MedQA-USMLE) under a deliberately tight
budget: **two Kaggle T4 GPUs (16 GB each), a few thousand labels, and 4-bit QLoRA**. The
project studies what cheap, post-hoc additions actually move the needle — and reports the
ones that do **and** the ones that don't.

## Headline results (Qwen2.5-7B student, 2,000 labels, MedQA-USMLE test split n=1,273)

| Component | Result | Notes |
|---|---|---|
| Supervised QLoRA student | **0.633** acc (95% CI [0.606, 0.660]) | from a 0.592 zero-shot base |
| Temperature calibration | ECE **0.086 → 0.026** | one scalar; most-confident 50% answered at 80.8% |
| **Order-invariant readout** (cyclic marginalization) | acc **0.633 → 0.655** | paired +0.022, 95% CI [+0.006, +0.038]; AURC 0.185→0.177; training-free, 4 forward passes |
| Option-ID prior removal (PriDe) | +0.000 (n.s.) | prior is near-uniform here — nothing to debias |
| Reflective routing (reason on ~2% least-confident) | acc **0.633 → 0.639** | paired +0.006, 95% CI [+0.002, +0.012]; ~2 extra tokens/q |
| **Negative:** trained reflect adapter | +0.002 (n.s.) | no better than the base student's own reasoning |
| **Negative:** teacher self-distillation | lowers val accuracy | not retained by the checkpoint rule |
| **Negative:** 5-sample self-consistency | 0.493 vs 0.709 | loses to a single calibrated pass (matched 134-item subset) |

The working recipe: **stronger base + light supervision + temperature calibration +
abstention + order-invariant readout.**

## Repository layout

```
paper/        LaTeX sources (CALM_paper.tex = consolidated; AI.tex = humanized submission),
              the architecture diagram (.drawio + .png), and paper figures
notebooks/    Kaggle-ready notebooks (2×T4):
                NOTE-RL_MedQA_v3_qwen7b.ipynb     – main 7B QLoRA training pipeline
                NOTE-RL_MedQA_v3_qwen_qlora.ipynb – 3B variant
                CALM_PI_MedQA_7b.ipynb            – order-invariant (cyclic-marginalized) readout
                CALM_R3_final.ipynb               – reflective routing (compute-aware, honest eval)
                CALM_R3_MedQA_7b.ipynb            – routing diagnostic (Phase 0)
src/          Stand-alone pipeline modules used by / extracted from the notebooks
results/      Metric artifacts (JSON/CSV) and plots from the reported runs
docs/         Supplementary notes and paper text blocks
```

## Reproducing the numbers

1. **Environment:** a Kaggle notebook with 2× T4 GPUs (16 GB each), internet on. Set your
   `HF_TOKEN` in Kaggle *Secrets* (no keys are hard-coded anywhere in this repo).
2. **Train the student:** run `notebooks/NOTE-RL_MedQA_v3_qwen7b.ipynb`. It saves a LoRA
   adapter (`best_student_adapter/`).
3. **Order-invariant readout:** run `notebooks/CALM_PI_MedQA_7b.ipynb`. Point `ADAPTER_SRC`
   at your trained adapter (or add it as a Kaggle Input; the loader auto-detects
   `best_student_adapter`). Produces `pi_results.json` (base vs PriDe vs cyclic-marginalized
   with bootstrap CIs). ~50–90 min, no token generation.
4. **Reflective routing:** run `notebooks/CALM_R3_final.ipynb`. It calibrates generation
   speed, sizes the run to your remaining budget, selects the routing threshold on
   validation, and reports paired CIs (`phaseA.json`, `phaseC.json`).

## Data and model weights

- **Data:** MedQA-USMLE (4-option). The notebooks load it directly; splits are the official
  dev/test files with a fixed seed (released with the code).
- **Trained adapter (~15 GB base + LoRA):** **not stored in this repo.** Host it on the
  Hugging Face Hub or as a Kaggle dataset and link it here:
  `ADAPTER: <add your HF/Kaggle URL>`. The evaluation notebooks accept a Hub id or a local
  `best_student_adapter` path.

## Paper

`paper/CALM_paper.tex` is the consolidated source; `paper/AI.tex` is the submission-ready
(human-edited) version. Build with any IEEEtran-capable LaTeX toolchain; figures are in
`paper/`. The diagram source is `paper/CALM_architecture.drawio` (export to
`architecture.png`).

## Notes

- A generative AI assistant was used for grammar/copy-editing and for scaffolding code; all
  experimental design, runs, analysis, and conclusions are the authors' own.
- License: add your choice (e.g. MIT) before making the repository public.
