# =====================================================================================
# CALM-R3 : Calibration-Aware Reflective Routing & Reasoning   (research-grade module)
# -------------------------------------------------------------------------------------
# Paste as NEW CELLS at the END of NOTE-RL_MedQA_v3_qwen7b.ipynb, AFTER training, with
# the validation-selected `student` checkpoint loaded. Reuses existing objects:
#   student, student_tok, STU_LETTER_IDS, STUDENT_DEVICE, OPTIONS, cfg, val_ds, test_ds,
#   seed_ds, format_prompt, stu_chat, _fwd_last, _letter_logits, student_option_logits,
#   _parse_letter, fit_temperature, expected_calibration_error, have_time.
#
# PHASE 0 = the VALIDATION (no training, cheap). It answers the only question that
#           matters: does the reasoning path beat the fast path on uncertain items?
# PHASE 1 = the architecture (two-adapter routing + corrected Reflective Correction
#           Loss). Run it ONLY if Phase 0 prints GO. Gated by RUN_PHASE1.
# All artifacts are written to cfg.out_dir and zipped at the end.
# =====================================================================================
import os, json, time, math, csv, numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R3_DIR = os.path.join(cfg.out_dir, "calm_r3"); os.makedirs(R3_DIR, exist_ok=True)

# ---- knobs (raise for the full run; defaults are budget-safe) ----
REASON_TEST_N   = 600          # test items to generate a rationale for (1273 = full split)
REASON_MAXTOK   = cfg.reason_cot_tokens   # 256 by default
ROUTE_GRID      = np.round(np.linspace(0.30, 0.95, 14), 3)  # fast-confidence thresholds tau
RUN_PHASE1      = False         # <-- set True ONLY after Phase 0 says GO


def _sm(z, T=1.0):
    z = np.asarray(z) / T; z = z - z.max(-1, keepdims=True)
    e = np.exp(z); return e / e.sum(-1, keepdims=True)

def _use_cache(flag):
    try: student.config.use_cache = flag
    except Exception: pass

# ---------- reasoning path (single deterministic chain, then read option logits) ----------
@torch.no_grad()
def gen_rationale(q, opts, max_new=REASON_MAXTOK):
    user = ("You are a medical expert. Think through the differential step by step, "
            "then end with 'Final Answer: X'.\nQuestion: " + q +
            "\nA. " + opts[0] + "\nB. " + opts[1] + "\nC. " + opts[2] + "\nD. " + opts[3])
    enc = student_tok([stu_chat(user)], return_tensors="pt", truncation=True,
                      max_length=cfg.max_src_len).to(STUDENT_DEVICE)
    gen = student.generate(**enc, do_sample=False, max_new_tokens=max_new,
                           pad_token_id=student_tok.pad_token_id)
    new = gen[0, enc.input_ids.shape[1]:]
    txt = student_tok.decode(new, skip_special_tokens=True)
    ntok = int((new != student_tok.pad_token_id).sum().item())
    del enc, gen; torch.cuda.empty_cache()
    return txt, ntok, _parse_letter(txt)

@torch.no_grad()
def reason_logits(q, opts, rationale):
    su = (format_prompt(q, opts) + "\n\nStep-by-step reasoning:\n" + rationale.strip() +
          "\n\nGiven the reasoning above, reply with ONLY one letter: A, B, C, or D.")
    enc = student_tok([stu_chat(su)], return_tensors="pt", truncation=True,
                      max_length=cfg.max_src_len + REASON_MAXTOK + 64).to(STUDENT_DEVICE)
    last = _fwd_last(student, enc.input_ids, enc.attention_mask)
    out = _letter_logits(last)[0].float().cpu().numpy()
    del enc, last; torch.cuda.empty_cache()
    return out


# =====================================================================================
# PHASE 0 -- ROUTING DIAGNOSTIC  (the validation)
# =====================================================================================
def phase0():
    import random
    print("\n" + "=" * 70 + "\nPHASE 0  CALM-R3 routing diagnostic (no training)\n" + "=" * 70)
    student.eval()

    # fast path: calibrate temperature on validation
    vlog = student_option_logits(student, [format_prompt(e["question"], e["options"]) for e in val_ds]).numpy()
    vgold = np.array([OPTIONS.index(e["answer"]) for e in val_ds])
    T = fit_temperature(vlog, vgold); print("fast-path temperature T=%.3f" % T)

    # choose a test subset
    rng = random.Random(cfg.seed); idx = list(range(len(test_ds))); rng.shuffle(idx)
    idx = idx[:min(REASON_TEST_N, len(test_ds))]
    sub = [test_ds[i] for i in idx]
    gold = np.array([OPTIONS.index(e["answer"]) for e in sub])

    # fast predictions + calibrated confidence
    flog = student_option_logits(student, [format_prompt(e["question"], e["options"]) for e in sub]).numpy()
    fp = _sm(flog, T); fast_pred = fp.argmax(1); fast_conf = fp.max(1)
    fast_correct = (fast_pred == gold).astype(int)

    # reasoning predictions (one chain each) + token cost
    _use_cache(True)
    r_pred = np.zeros(len(sub), int); r_tok = np.zeros(len(sub)); r_ok_parse = np.zeros(len(sub), int)
    t0 = time.time()
    for j, e in enumerate(sub):
        if not have_time(600):
            print("  [budget] stopping reasoning gen at", j);
            r_pred = r_pred[:j]; r_tok = r_tok[:j]; gold = gold[:j]
            fast_pred = fast_pred[:j]; fast_conf = fast_conf[:j]; fast_correct = fast_correct[:j]; break
        rat, ntok, letter = gen_rationale(e["question"], e["options"])
        if letter not in OPTIONS:   # fall back to logit read over the rationale
            letter = OPTIONS[int(reason_logits(e["question"], e["options"], rat).argmax())]
        r_pred[j] = OPTIONS.index(letter); r_tok[j] = ntok
        if (j + 1) % 50 == 0: print("  reasoned %d/%d  (%.1f min)" % (j + 1, len(sub), (time.time() - t0) / 60))
    _use_cache(False)
    n = len(gold); reason_correct = (r_pred == gold).astype(int)

    fast_acc = float(fast_correct.mean()); reason_acc = float(reason_correct.mean())
    mean_rtok = float(r_tok[:n].mean())
    print("\nsubset n=%d | fast acc=%.4f | reason acc=%.4f | mean reason tokens=%.1f"
          % (n, fast_acc, reason_acc, mean_rtok))

    # ---- per-decile diagnostic: does reasoning beat fast where fast is unsure? ----
    order = np.argsort(fast_conf); buckets = np.array_split(order, 10)
    rows = []
    for bi, b in enumerate(buckets):
        rows.append(dict(decile=bi + 1, n=len(b),
                         conf_lo=float(fast_conf[b].min()), conf_hi=float(fast_conf[b].max()),
                         fast_acc=float(fast_correct[b].mean()), reason_acc=float(reason_correct[b].mean()),
                         reason_minus_fast=float(reason_correct[b].mean() - fast_correct[b].mean())))
    with open(os.path.join(R3_DIR, "phase0_buckets.csv"), "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    print("\n decile  conf-range      fast   reason   reason-fast")
    for r in rows:
        print("  %2d  [%.2f,%.2f]   %.3f   %.3f    %+.3f"
              % (r["decile"], r["conf_lo"], r["conf_hi"], r["fast_acc"], r["reason_acc"], r["reason_minus_fast"]))

    # ---- threshold sweep: route conf<tau to reasoning; accuracy vs mean tokens/q ----
    sweep = []
    for tau in ROUTE_GRID:
        route = fast_conf < tau
        pred = np.where(route, r_pred, fast_pred)
        acc = float((pred == gold).mean()); frac = float(route.mean())
        toks = float(np.where(route, r_tok[:n], 0.0).mean())   # fast path generates ~0 tokens
        sweep.append(dict(tau=float(tau), routed_acc=acc, frac_reasoned=frac, mean_tokens=toks,
                          ace=acc / (toks + 1.0)))
    with open(os.path.join(R3_DIR, "phase0_sweep.csv"), "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(sweep[0].keys())); w.writeheader(); [w.writerow(r) for r in sweep]
    best = max(sweep, key=lambda r: r["routed_acc"])

    # ---- GO / NO-GO verdict ----
    low = np.concatenate([buckets[0], buckets[1], buckets[2]])   # bottom 30% confidence
    reason_helps_low = float(reason_correct[low].mean() - fast_correct[low].mean())
    go = (best["routed_acc"] > fast_acc + 0.005) and (reason_helps_low > 0)
    verdict = "GO" if go else "NO-GO"
    print("\n" + "-" * 70)
    print("best routed acc = %.4f @ tau=%.2f (%.0f%% reasoned, %.1f tok/q)"
          % (best["routed_acc"], best["tau"], 100 * best["frac_reasoned"], best["mean_tokens"]))
    print("fast-only acc   = %.4f   |   reason-only acc = %.4f" % (fast_acc, reason_acc))
    print("reason - fast on bottom-30%% confidence = %+.4f" % reason_helps_low)
    print("VERDICT: %s  -- %s" % (verdict,
          "routing helps; Phase 1 training is justified." if go else
          "reasoning does NOT beat fast where it matters; do NOT train Phase 1 for this student."))
    print("-" * 70)

    # ---- plots ----
    plt.figure(figsize=(5, 3.4))
    xs = [r["decile"] for r in rows]
    plt.plot(xs, [r["fast_acc"] for r in rows], "o-", label="fast")
    plt.plot(xs, [r["reason_acc"] for r in rows], "s-", label="reason")
    plt.xlabel("fast-confidence decile (1=least sure)"); plt.ylabel("accuracy")
    plt.title("Fast vs reasoning by confidence"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(R3_DIR, "phase0_bucket_acc.png"), dpi=200); plt.close()

    plt.figure(figsize=(5, 3.4))
    plt.plot([r["mean_tokens"] for r in sweep], [r["routed_acc"] for r in sweep], "o-")
    plt.scatter([0], [fast_acc], c="k", marker="x", s=60, label="always-fast")
    plt.scatter([mean_rtok], [reason_acc], c="r", marker="*", s=90, label="always-reason")
    plt.xlabel("mean generated tokens / question"); plt.ylabel("routed accuracy")
    plt.title("Accuracy vs compute (CALM-R3 frontier)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(R3_DIR, "phase0_acc_vs_compute.png"), dpi=200); plt.close()

    out = dict(n=n, T_fast=T, fast_acc=fast_acc, reason_acc=reason_acc, mean_reason_tokens=mean_rtok,
               best=best, reason_helps_bottom30=reason_helps_low, verdict=verdict, buckets=rows, sweep=sweep)
    with open(os.path.join(R3_DIR, "phase0_results.json"), "w") as f: json.dump(out, f, indent=2)
    print("saved Phase 0 artifacts ->", R3_DIR)
    return out

phase0_out = phase0()


# =====================================================================================
# PHASE 1 -- TWO-ADAPTER ROUTING + corrected Reflective Correction Loss  (gated)
#   Fast path  = existing trained adapter ("default").
#   Reflect path = new "reflect" LoRA adapter, trained with:
#     L = CE(reflect, y) on HARD items (fast wrong / unsure)               [teach correction]
#       + mu * KL( stopgrad softmax(fast) || softmax(reflect) ) on EASY    [preserve instinct]
#   Reasoning is conditioned on an OFFLINE rationale (no in-loop generation -> budget-safe).
# =====================================================================================
HARD_N, EASY_N = 400, 200
RFT_EPOCHS, RFT_LR, RFT_BS, RFT_ACC, RFT_MU = 2, 1e-4, 2, 8, 0.3
RAT_TOK = 160  # rationale length used for training prompts


def _adapter(name):
    try: student.set_adapter(name)
    except Exception as e: print("  set_adapter(%s) warn: %s" % (name, e))

@torch.no_grad()
def _offline_rationale(q, opts):
    _use_cache(True)
    txt, _, _ = gen_rationale(q, opts, max_new=RAT_TOK)
    _use_cache(False)
    return txt

def _score_prompt(q, opts, rationale):
    return (format_prompt(q, opts) + "\n\nStep-by-step reasoning:\n" + rationale.strip() +
            "\n\nGiven the reasoning above, reply with ONLY one letter: A, B, C, or D.")

def _letter_logits_grad(prompts):
    chats = [stu_chat(p) for p in prompts]
    enc = student_tok(chats, return_tensors="pt", padding=True, truncation=True,
                      max_length=cfg.max_src_len + RAT_TOK + 64).to(STUDENT_DEVICE)
    last = _fwd_last(student, enc.input_ids, enc.attention_mask)   # grad ON
    return _letter_logits(last)

def phase1():
    from peft import LoraConfig
    import random
    print("\n" + "=" * 70 + "\nPHASE 1  two-adapter routing + Reflective Correction Loss\n" + "=" * 70)
    student.eval(); _adapter("default")

    # ---- split seed_ds into EASY (fast right & confident) and HARD (fast wrong / unsure) ----
    T = phase0_out["T_fast"]
    sp = [format_prompt(e["question"], e["options"]) for e in seed_ds]
    slog = student_option_logits(student, sp).numpy(); sP = _sm(slog, T)
    sg = np.array([OPTIONS.index(e["answer"]) for e in seed_ds])
    spred = sP.argmax(1); sconf = sP.max(1); scorrect = spred == sg
    hard_i = [i for i in range(len(seed_ds)) if (not scorrect[i]) or sconf[i] < 0.5][:HARD_N]
    easy_i = [i for i in range(len(seed_ds)) if scorrect[i] and sconf[i] >= 0.7][:EASY_N]
    print("hard=%d easy=%d" % (len(hard_i), len(easy_i)))

    # ---- offline rationales + training examples ----
    _use_cache(True)
    ex = []   # (score_prompt, gold_idx, type, soft_target_or_None)
    for k, i in enumerate(hard_i):
        if not have_time(700): print("  [budget] stop hard-rationale at", k); break
        e = seed_ds[i]; rat = _offline_rationale(e["question"], e["options"])
        ex.append((_score_prompt(e["question"], e["options"], rat), int(sg[i]), "hard", None))
    for k, i in enumerate(easy_i):
        if not have_time(700): print("  [budget] stop easy-rationale at", k); break
        e = seed_ds[i]; rat = _offline_rationale(e["question"], e["options"])
        ex.append((_score_prompt(e["question"], e["options"], rat), int(sg[i]),
                   "easy", sP[i].astype(np.float32)))
    _use_cache(False)
    random.Random(cfg.seed).shuffle(ex)
    print("training examples:", len(ex))

    # ---- add + activate the reflect adapter; train only its params ----
    rcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      bias="none", task_type="CAUSAL_LM", target_modules=list(cfg.lora_targets))
    try: student.add_adapter("reflect", rcfg)
    except Exception as e: print("  add_adapter warn (may already exist):", e)
    _adapter("reflect")
    for n_, p_ in student.named_parameters():
        p_.requires_grad = ("lora" in n_ and "reflect" in n_)
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=RFT_LR); _use_cache(False); student.train()
    print("reflect trainable tensors:", len(params))

    steps = 0
    for ep in range(RFT_EPOCHS):
        for b in range(0, len(ex), RFT_BS):
            if not have_time(600): print("  [budget] stop training"); break
            batch = ex[b:b + RFT_BS]
            logits = _letter_logits_grad([z[0] for z in batch]).float()
            gold = torch.tensor([z[1] for z in batch], device=logits.device)
            mask_hard = torch.tensor([z[2] == "hard" for z in batch], device=logits.device)
            loss = torch.tensor(0.0, device=logits.device)
            if mask_hard.any():
                loss = loss + F.cross_entropy(logits[mask_hard], gold[mask_hard])
            me = ~mask_hard
            if me.any():
                soft = torch.tensor(np.stack([z[3] for z in batch if z[2] == "easy"]),
                                    device=logits.device).float()
                logp = F.log_softmax(logits[me], -1)
                loss = loss + RFT_MU * F.kl_div(logp, soft, reduction="batchmean")
            (loss / RFT_ACC).backward(); steps += 1
            if steps % RFT_ACC == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip); opt.step(); opt.zero_grad()
        print("  epoch %d done (loss=%.4f)" % (ep + 1, float(loss)))
    student.eval(); _use_cache(False)
    save_dir = os.path.join(R3_DIR, "reflect_adapter")
    student.save_pretrained(save_dir)  # writes all adapters incl. reflect
    print("saved reflect adapter ->", save_dir)

    # ---- routed evaluation on a test subset ----
    tau = phase0_out["best"]["tau"]
    rng = random.Random(cfg.seed + 1); idx = list(range(len(test_ds))); rng.shuffle(idx)
    idx = idx[:min(REASON_TEST_N, len(test_ds))]; sub = [test_ds[i] for i in idx]
    gold = np.array([OPTIONS.index(e["answer"]) for e in sub])
    _adapter("default")
    flog = student_option_logits(student, [format_prompt(e["question"], e["options"]) for e in sub]).numpy()
    fp = _sm(flog, T); fast_pred = fp.argmax(1); fast_conf = fp.max(1)
    route = fast_conf < tau
    pred = fast_pred.copy(); toks = np.zeros(len(sub))
    _use_cache(True)
    for j, e in enumerate(sub):
        if route[j]:
            if not have_time(500): break
            _adapter("reflect"); rat, ntok, letter = gen_rationale(e["question"], e["options"])
            if letter not in OPTIONS:
                letter = OPTIONS[int(reason_logits(e["question"], e["options"], rat).argmax())]
            pred[j] = OPTIONS.index(letter); toks[j] = ntok; _adapter("default")
    _use_cache(False)
    routed_acc = float((pred == gold).mean()); fast_acc = float((fast_pred == gold).mean())
    res = dict(tau=tau, routed_acc=routed_acc, fast_only_acc=fast_acc,
               frac_reasoned=float(route.mean()), mean_tokens=float(toks.mean()),
               delta_vs_fast=routed_acc - fast_acc)
    with open(os.path.join(R3_DIR, "phase1_routed.json"), "w") as f: json.dump(res, f, indent=2)
    print("ROUTED: acc=%.4f (fast-only %.4f, %+.4f) | %.0f%% reasoned | %.1f tok/q"
          % (routed_acc, fast_acc, routed_acc - fast_acc, 100 * route.mean(), toks.mean()))
    return res

if RUN_PHASE1 and phase0_out["verdict"] == "GO":
    phase1_out = phase1()
else:
    print("\nPhase 1 skipped (RUN_PHASE1=%s, verdict=%s)." % (RUN_PHASE1, phase0_out["verdict"]))


# =====================================================================================
# ZIP all artifacts
# =====================================================================================
import shutil
zip_base = os.path.join(cfg.out_dir, "calm_r3_artifacts")
shutil.make_archive(zip_base, "zip", R3_DIR)
print("\nzipped ->", zip_base + ".zip")
print("contents:", sorted(os.listdir(R3_DIR)))

