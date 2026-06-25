# =====================================================================================
# CALM permutation-invariant calibrated readout  (training-free, test-time)
#   Compares on the trained 7B student, with bootstrap CIs (honest protocol):
#     (a) BASE     : canonical single-pass calibrated readout            (1 forward/item)
#     (b) PriDe    : option-ID prior debiasing                           (~1 forward/item)
#     (c) CYC-MARG : 4-way cyclic-permutation marginalization (ours)     (4 forward/item)
#   Reports acc (+CI), ECE, AURC, cov@80, per-shift accuracy spread (robustness),
#   and paired CIs of (method - base). No token generation -> fast & cheap.
# Requires infra: student, student_tok, OPTIONS, cfg, val_ds, test_ds, format_prompt,
#   student_option_logits, expected_calibration_error, fit_temperature.
# =====================================================================================
import os, json, csv, time, numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PI = os.path.join(cfg.out_dir, "calm_pi"); os.makedirs(PI, exist_ok=True)
SHIFTS    = (0, 1, 2, 3)
VAL_FIT_N = 500     # val items for temperature + PriDe prior (None = all 1000)
BOOT_N    = 2000


def _sm(z, T=1.0):
    z = np.asarray(z, float) / T; z = z - z.max(-1, keepdims=True)
    e = np.exp(z); return e / e.sum(-1, keepdims=True)

def content_probs_over_shifts(ds, shifts=SHIFTS):
    """Evaluate all cyclic shifts; read per-LETTER softmax back to per-CONTENT prob.
    Under shift s, displayed letter j shows original content (j+s)%4, so content c
    sits at letter (c-s)%4. Returns P_marg[N,4], golds[N], base_logits[N,4](s=0),
    letter_probs[S,N,4]."""
    N = len(ds); prompts = []
    for s in shifts:
        for i in range(N):
            opts = ds[i]["options"]; disp = [opts[(j + s) % 4] for j in range(4)]
            prompts.append(format_prompt(ds[i]["question"], disp))
    t0 = time.time()
    logits = student_option_logits(student, prompts).numpy().reshape(len(shifts), N, 4)
    print("    %d forward prompts in %.1f min" % (len(prompts), (time.time() - t0) / 60))
    letter_probs = _sm(logits)
    content = np.zeros_like(letter_probs)
    for si, s in enumerate(shifts):
        for c in range(4):
            content[si, :, c] = letter_probs[si, :, (c - s) % 4]
    P = content.mean(0)
    golds = np.array([OPTIONS.index(ds[i]["answer"]) for i in range(N)])
    return P, golds, logits[0], letter_probs

def aurc(correct, conf):
    o = np.argsort(-conf); c = np.asarray(correct)[o]
    return float(np.trapz(1 - np.cumsum(c) / np.arange(1, len(c) + 1), np.arange(1, len(c) + 1) / len(c)))
def cov_at_acc(correct, conf, tgt=0.80):
    o = np.argsort(-conf); c = np.asarray(correct)[o]; run = np.cumsum(c) / np.arange(1, len(c) + 1)
    ok = np.where(run >= tgt)[0]; return float((ok.max() + 1) / len(c)) if len(ok) else 0.0
def boot_acc(correct, n=BOOT_N, seed=1):
    c = np.asarray(correct, float); N = len(c); rng = np.random.default_rng(seed)
    a = np.array([c[rng.integers(0, N, N)].mean() for _ in range(n)])
    return float(c.mean()), (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
def boot_diff(a, b, n=BOOT_N, seed=0):
    a = np.asarray(a, float); b = np.asarray(b, float); N = len(a); rng = np.random.default_rng(seed)
    d = np.array([(a[s].mean() - b[s].mean()) for s in (rng.integers(0, N, N) for _ in range(n))])
    return float(a.mean() - b.mean()), (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))

def report(name, P, golds):
    pred = P.argmax(1); conf = P.max(1); correct = (pred == golds).astype(int)
    acc, ci = boot_acc(correct)
    ece = expected_calibration_error(correct, conf, cfg.ece_bins)
    oh = np.eye(4)[golds]; brier = float(np.mean(np.sum((P - oh) ** 2, 1)))
    a = aurc(correct, conf); cov = cov_at_acc(correct, conf, 0.80)
    print("[%-16s] acc=%.4f CI[%.4f,%.4f]  ECE=%.4f  Brier=%.3f  AURC=%.4f  cov@80=%.3f"
          % (name, acc, ci[0], ci[1], ece, brier, a, cov))
    return dict(acc=acc, acc_ci=ci, ece=ece, brier=brier, aurc=a, cov80=cov), correct, conf


print("\n" + "=" * 72 + "\nCALM permutation-invariant calibrated readout\n" + "=" * 72)
student.eval()
valfit = [val_ds[i] for i in range(min(VAL_FIT_N or len(val_ds), len(val_ds)))]
print(">> validation shifts (for temperature + prior) ...")
Pv, gv, baseLv, lpv = content_probs_over_shifts(valfit)
print(">> test shifts (full split) ...")
Pt, gt, baseLt, lpt = content_probs_over_shifts(test_ds)

print("\n--- TEST (n=%d) ---" % len(gt))
# (a) BASE calibrated
Tb = fit_temperature(baseLv, gv)
r_base, c_base, _ = report("BASE cal (T=%.2f)" % Tb, _sm(baseLt, Tb), gt)
# (b) PriDe
prior = lpv.reshape(-1, 4).mean(0); prior = prior / prior.sum()
canon_v = _sm(baseLv); canon_t = _sm(baseLt)
prideLv = np.log(np.clip(canon_v / prior[None, :], 1e-9, 1)); prideLt = np.log(np.clip(canon_t / prior[None, :], 1e-9, 1))
Tp = fit_temperature(prideLv, gv)
r_pride, c_pride, _ = report("PriDe cal (T=%.2f)" % Tp, _sm(prideLt, Tp), gt)
print("    option-ID prior A/B/C/D =", np.round(prior, 3).tolist())
# (c) CYC-MARG (ours)
cycLv = np.log(np.clip(Pv, 1e-9, 1)); cycLt = np.log(np.clip(Pt, 1e-9, 1))
Tc = fit_temperature(cycLv, gv)
r_cyc, c_cyc, conf_cyc = report("CYC-MARG (T=%.2f)" % Tc, _sm(cycLt, Tc), gt)

# paired gains vs base
d_pride, ci_pride = boot_diff(c_pride, c_base)
d_cyc, ci_cyc = boot_diff(c_cyc, c_base)
print("\nPriDe   - base = %+.4f  CI[%+.4f,%+.4f]  -> %s" % (d_pride, *ci_pride, "significant" if ci_pride[0] > 0 else "n.s."))
print("CYC-MARG- base = %+.4f  CI[%+.4f,%+.4f]  -> %s" % (d_cyc, *ci_cyc, "significant" if ci_cyc[0] > 0 else "n.s."))

# robustness: per-shift base accuracy spread (cyc-marg is invariant by construction)
per_shift = []
for si in range(lpt.shape[0]):
    pc = np.zeros((len(gt), 4))
    for c in range(4): pc[:, c] = lpt[si, :, (c - SHIFTS[si]) % 4]
    per_shift.append(float((pc.argmax(1) == gt).mean()))
print("\nper-shift BASE acc:", [f"{a:.3f}" for a in per_shift],
      "  spread=%.3f  std=%.4f" % (max(per_shift) - min(per_shift), float(np.std(per_shift))))
print("CYC-MARG is permutation-invariant by construction (per-question prediction is order-independent).")

# save
out = {"n_test": len(gt), "T_base": Tb, "T_pride": Tp, "T_cyc": Tc,
       "base": r_base, "pride": r_pride, "cyc_marg": r_cyc,
       "pride_minus_base": d_pride, "pride_ci": ci_pride,
       "cyc_minus_base": d_cyc, "cyc_ci": ci_cyc,
       "id_prior_ABCD": prior.tolist(), "per_shift_base_acc": per_shift,
       "per_shift_spread": float(max(per_shift) - min(per_shift)),
       "per_shift_std": float(np.std(per_shift))}
def _ser(o):
    if isinstance(o, dict): return {k: _ser(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_ser(v) for v in o]
    return o
json.dump(_ser(out), open(os.path.join(PI, "pi_results.json"), "w"), indent=2)

# plots
plt.figure(figsize=(5, 3.4))
names = ["base", "PriDe", "cyc-marg"]; accs = [r_base["acc"], r_pride["acc"], r_cyc["acc"]]
errs = [[a - lo for a, (lo, hi) in [(r_base["acc"], r_base["acc_ci"]), (r_pride["acc"], r_pride["acc_ci"]), (r_cyc["acc"], r_cyc["acc_ci"])]],
        [hi - a for a, (lo, hi) in [(r_base["acc"], r_base["acc_ci"]), (r_pride["acc"], r_pride["acc_ci"]), (r_cyc["acc"], r_cyc["acc_ci"])]]]
plt.bar(names, accs, yerr=errs, capsize=5, color=["#888", "#5a9", "#36c"])
plt.ylim(min(accs) - 0.03, max(accs) + 0.03); plt.ylabel("test accuracy"); plt.title("Readout accuracy (95% CI)")
plt.tight_layout(); plt.savefig(os.path.join(PI, "pi_accuracy.png"), dpi=200); plt.close()

def _rc(correct, conf):
    o = np.argsort(-conf); c = np.asarray(correct)[o]
    return np.arange(1, len(c) + 1) / len(c), 1 - np.cumsum(c) / np.arange(1, len(c) + 1)
plt.figure(figsize=(5, 3.4))
cb, rb = _rc(c_base, _sm(baseLt, Tb).max(1)); cc, rc = _rc(c_cyc, conf_cyc)
plt.plot(cb, rb, label="base"); plt.plot(cc, rc, label="cyc-marg (ours)")
plt.xlabel("coverage"); plt.ylabel("risk (error)"); plt.title("Risk–coverage"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(PI, "pi_risk_coverage.png"), dpi=200); plt.close()

import shutil
zb = os.path.join(cfg.out_dir, "calm_pi_artifacts")
shutil.make_archive(zb, "zip", PI)
print("\nzipped ->", zb + ".zip", "| contents:", sorted(os.listdir(PI)))
print("\nSEND ME: pi_results.json  (or the BASE / PriDe / CYC-MARG lines + the two paired CIs).")
