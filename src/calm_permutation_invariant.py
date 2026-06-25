# =====================================================================================
# CALM-PI : Permutation-Invariant Calibrated Readout  (test-time, TRAINING-FREE)
# -------------------------------------------------------------------------------------
# Paste this as a NEW CELL at the END of NOTE-RL_MedQA_v3_qwen7b.ipynb, AFTER the run
# has finished and the validation-selected `student` checkpoint is loaded.
# It does NOT retrain anything. It only does extra forward passes, so it is cheap
# (~4 forward passes per item, no decoding). Expect ~20-45 min on one T4 for the 7B.
#
# It compares three readouts on the SAME trained student:
#   (a) BASE      : canonical single-pass calibrated readout      (paper headline, 1x cost)
#   (b) PriDe     : single-pass option-ID prior debiasing         (~1x cost)  [Zheng+ ICLR'24]
#   (c) CYC-MARG  : 4-way cyclic-permutation marginalization      (4x cost)   [ours, the method]
#
# Reuses existing notebook objects: student, student_tok, val_ds, test_ds,
#   OPTIONS, cfg, format_prompt(), student_option_logits(),
#   expected_calibration_error(), fit_temperature().
# =====================================================================================
import os, json, time, numpy as np, torch, torch.nn.functional as F

MODEL      = student          # trained, validation-selected PEFT student
VAL, TEST  = val_ds, test_ds  # lists of {"question","options"(len4),"answer"(letter)}
SHIFTS     = (0, 1, 2, 3)     # cyclic shifts of the 4-option block

# Optionally subsample validation to fit the temperature faster (set None to use all):
VAL_FIT_N  = None             # e.g. 600 to save a few minutes; None = use full val split


def _softmax_np(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def content_probs_over_shifts(ds, shifts=SHIFTS):
    """For every item, evaluate all cyclic shifts of the option block and read each
    shift's per-LETTER softmax back into per-CONTENT probabilities.
    Convention: under shift s, displayed letter j shows original content (j+s) % 4,
    so content c sits at letter (c - s) % 4.
    Returns:
        P_marg : [N,4] content prob averaged over shifts (permutation-invariant)
        golds  : [N]   gold content index
        base_logits : [N,4] canonical (s=0) letter logits  == paper's base readout
        letter_probs: [S,N,4] per-shift letter softmax (for the prior estimate + per-shift acc)
    """
    N = len(ds)
    prompts = []
    for s in shifts:
        for i in range(N):
            ex = ds[i]; opts = ex["options"]
            disp = [opts[(j + s) % 4] for j in range(4)]
            prompts.append(format_prompt(ex["question"], disp))
    t0 = time.time()
    logits = student_option_logits(MODEL, prompts).numpy()        # [S*N, 4]
    logits = logits.reshape(len(shifts), N, 4)
    letter_probs = _softmax_np(logits)                            # softmax over displayed letters
    content = np.zeros_like(letter_probs)
    for si, s in enumerate(shifts):
        for c in range(4):
            content[si, :, c] = letter_probs[si, :, (c - s) % 4]
    P_marg = content.mean(0)                                      # [N,4]
    golds  = np.array([OPTIONS.index(ds[i]["answer"]) for i in range(N)])
    base_logits = logits[0]                                       # canonical order (s=0)
    print("  forward passes: %d prompts in %.1f min" % (len(prompts), (time.time() - t0) / 60))
    return P_marg, golds, base_logits, letter_probs


# ---------- metric helpers ----------
def aurc(correct, conf):
    order = np.argsort(-conf); c = np.asarray(correct)[order]
    cov  = np.arange(1, len(c) + 1) / len(c)
    risk = 1.0 - np.cumsum(c) / np.arange(1, len(c) + 1)
    return float(np.trapz(risk, cov))

def cov_at_acc(correct, conf, tgt=0.80):
    order = np.argsort(-conf); c = np.asarray(correct)[order]
    run = np.cumsum(c) / np.arange(1, len(c) + 1)
    ok = np.where(run >= tgt)[0]
    return float((ok.max() + 1) / len(c)) if len(ok) else 0.0

def report(name, P, golds):
    preds = P.argmax(1); conf = P.max(1); correct = (preds == golds).astype(int)
    acc = float(correct.mean())
    ece = expected_calibration_error(correct, conf, cfg.ece_bins)
    oh  = np.eye(4)[golds]
    brier = float(np.mean(np.sum((P - oh) ** 2, 1)))
    nll = float(-np.mean(np.log(np.clip(P[np.arange(len(golds)), golds], 1e-12, 1.0))))
    a = aurc(correct, conf); cov = cov_at_acc(correct, conf, 0.80)
    print(f"[{name:18s}] acc={acc:.4f}  ECE={ece:.4f}  Brier={brier:.3f}  "
          f"NLL={nll:.3f}  AURC={a:.4f}  cov@80={cov:.3f}")
    return dict(acc=acc, ece=ece, brier=brier, nll=nll, aurc=a, cov80=cov)


# ============================ run ============================
print(">> gathering shifted forward passes on validation ...")
Pv, gv, baseLv, lpv = content_probs_over_shifts(VAL)
print(">> gathering shifted forward passes on test ...")
Pt, gt, baseLt, lpt = content_probs_over_shifts(TEST)

fit_idx = slice(None) if VAL_FIT_N is None else slice(0, VAL_FIT_N)

print("\n================ RESULTS (test split) ================")

# (a) BASE -- canonical single pass
report("BASE raw (T=1)", _softmax_np(baseLt), gt)
Tb = fit_temperature(baseLv[fit_idx], gv[fit_idx])
r_base = report(f"BASE cal (T={Tb:.2f})", _softmax_np(baseLt / Tb), gt)

# (b) PriDe -- estimate option-ID prior from validation permutations, divide it out
prior = lpv.reshape(-1, 4).mean(0); prior = prior / prior.sum()
canon_v = _softmax_np(baseLv); canon_t = _softmax_np(baseLt)
prideLv = np.log(np.clip(canon_v / prior[None, :], 1e-9, 1.0))
prideLt = np.log(np.clip(canon_t / prior[None, :], 1e-9, 1.0))
Tp = fit_temperature(prideLv[fit_idx], gv[fit_idx])
r_pride = report(f"PriDe cal (T={Tp:.2f})", _softmax_np(prideLt / Tp), gt)
print("  estimated option-ID prior A/B/C/D =", np.round(prior, 3).tolist())

# (c) CYC-MARG -- our permutation-invariant readout
cycLv = np.log(np.clip(Pv, 1e-9, 1.0)); cycLt = np.log(np.clip(Pt, 1e-9, 1.0))
Tc = fit_temperature(cycLv[fit_idx], gv[fit_idx])
r_cyc = report(f"CYC-MARG cal (T={Tc:.2f})", _softmax_np(cycLt / Tc), gt)

# ---- robustness: accuracy under each individual cyclic shift (base) vs cyc-marg ----
per_shift = []
for si in range(lpt.shape[0]):
    pc = np.zeros((len(gt), 4))
    for c in range(4):
        pc[:, c] = lpt[si, :, (c - SHIFTS[si]) % 4]
    per_shift.append(float((pc.argmax(1) == gt).mean()))
print("\nper-shift base acc:", [f"{a:.3f}" for a in per_shift],
      "  spread(min->max)=%.3f" % (max(per_shift) - min(per_shift)),
      "  std=%.4f" % float(np.std(per_shift)))
print("CYC-MARG is permutation-invariant by construction (per-question prediction "
      "does not depend on the displayed order).")

# ---- save ----
out = {"base_cal": r_base, "pride_cal": r_pride, "cyc_marg_cal": r_cyc,
       "T_base": Tb, "T_pride": Tp, "T_cyc": Tc,
       "id_prior_ABCD": prior.tolist(),
       "per_shift_base_acc": per_shift,
       "per_shift_acc_std": float(np.std(per_shift)),
       "per_shift_acc_spread": float(max(per_shift) - min(per_shift))}
with open(os.path.join(cfg.out_dir, "calm_pi_results.json"), "w") as f:
    json.dump(out, f, indent=2)
print("\nsaved ->", os.path.join(cfg.out_dir, "calm_pi_results.json"))
print("\nPASTE the four lines below to me and I will wire the real numbers into the paper:")
print("  BASE :", r_base)
print("  PriDe:", r_pride)
print("  CYC  :", r_cyc)
print("  base per-shift acc std:", out["per_shift_acc_std"])
