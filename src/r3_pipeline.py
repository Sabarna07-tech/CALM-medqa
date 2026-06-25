# =====================================================================================
# CALM-R3 FINAL pipeline  (compute-aware, honest evaluation)
#   Phase A : honest diagnostic on UNTRAINED reasoning
#             - batched reasoning, routing-zone only
#             - threshold tau selected on VALIDATION, reported on TEST
#             - bootstrap CIs on (routed - fast)
#   Phase B : train the reflect adapter (corrected reflective-correction loss)
#   Phase C : routed evaluation with the TRAINED reflect adapter (+ CIs, ACE frontier)
#   Budget-adaptive: calibrates gen speed, caps N to fit HOURS_LEFT.
# Requires infra already defined: student, student_tok, STU_LETTER_IDS, STUDENT_DEVICE,
#   OPTIONS, cfg, val_ds, test_ds, seed_ds, format_prompt, stu_chat, _fwd_last,
#   _letter_logits, student_option_logits, fit_temperature, expected_calibration_error,
#   _parse_letter, have_time.
# =====================================================================================
import os, json, csv, time, numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R3 = os.path.join(cfg.out_dir, "calm_r3_final"); os.makedirs(R3, exist_ok=True)

# ----------------------------- knobs -----------------------------
HOURS_LEFT     = 11.0        # remaining Kaggle budget (edit me)
RESERVE_H      = 1.0
ROUTE_MAX      = 0.75        # never reason on items more confident than this
TAU_GRID       = np.round(np.arange(0.30, ROUTE_MAX + 1e-9, 0.025), 3)
GEN_BS         = 8           # batched reasoning generation
REASON_MAXTOK  = 192
MIN_REASON_TOK = 64
N_VAL_TAU      = 500         # val items used to choose tau
N_TEST         = 1273        # full test split (auto-capped to fit budget)
BOOT_N         = 2000
# Phase B
RUN_PHASE1     = True
HARD_N, EASY_N = 500, 250
RFT_EPOCHS, RFT_LR, RFT_BS, RFT_ACC, RFT_MU = 2, 1e-4, 2, 8, 0.3
RAT_TOK        = 160


def _use_cache(f):
    try: student.config.use_cache = f
    except Exception: pass

def _sm(z, T=1.0):
    z = np.asarray(z, float) / T; z = z - z.max(-1, keepdims=True)
    e = np.exp(z); return e / e.sum(-1, keepdims=True)

def _reason_user(q, opts):
    return ("You are an attending physician. Reason about THIS case BEFORE answering. "
            "In 3-5 sentences, compare the options and state the key differentiating "
            "feature; do NOT reveal the letter until the very end. "
            "Finish with a separate final line exactly: 'Final Answer: X'.\nQuestion: " + q +
            "\nA. " + opts[0] + "\nB. " + opts[1] + "\nC. " + opts[2] + "\nD. " + opts[3])

@torch.no_grad()
def batch_reason(items, bs=GEN_BS, max_new=REASON_MAXTOK, min_new=MIN_REASON_TOK):
    """items: list[(q,opts)] -> list[(text, n_tokens, parsed_letter)] (batched generation)."""
    _use_cache(True); student.eval(); out = [None] * len(items)
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        prompts = [stu_chat(_reason_user(q, o)) for (q, o) in chunk]
        enc = student_tok(prompts, return_tensors="pt", padding=True, truncation=True,
                          max_length=cfg.max_src_len).to(STUDENT_DEVICE)
        gen = student.generate(**enc, do_sample=False, min_new_tokens=min_new,
                               max_new_tokens=max_new, pad_token_id=student_tok.pad_token_id)
        new = gen[:, enc.input_ids.shape[1]:]
        for j in range(len(chunk)):
            row = new[j]; txt = student_tok.decode(row, skip_special_tokens=True)
            ntok = int((row != student_tok.pad_token_id).sum().item())
            out[i + j] = (txt, ntok, _parse_letter(txt))
        del enc, gen, new; torch.cuda.empty_cache()
    _use_cache(False); return out

@torch.no_grad()
def reason_logits(q, opts, rationale):
    su = (format_prompt(q, opts) + "\n\nStep-by-step reasoning:\n" + rationale.strip() +
          "\n\nGiven the reasoning above, reply with ONLY one letter: A, B, C, or D.")
    enc = student_tok([stu_chat(su)], return_tensors="pt", truncation=True,
                      max_length=cfg.max_src_len + REASON_MAXTOK + 64).to(STUDENT_DEVICE)
    out = _letter_logits(_fwd_last(student, enc.input_ids, enc.attention_mask))[0].float().cpu().numpy()
    del enc; torch.cuda.empty_cache(); return out

def reason_preds(items):
    """returns (pred_idx[N], tokens[N]) using batched generation + logit fallback on parse-fail."""
    res = batch_reason(items)
    pred = np.zeros(len(items), int); tok = np.zeros(len(items))
    for j, (txt, ntok, letter) in enumerate(res):
        if letter not in OPTIONS:
            letter = OPTIONS[int(reason_logits(items[j][0], items[j][1], txt).argmax())]
        pred[j] = OPTIONS.index(letter); tok[j] = ntok
    return pred, tok

def boot_diff(a, b, n=BOOT_N, seed=0):
    a = np.asarray(a, float); b = np.asarray(b, float); N = len(a)
    rng = np.random.default_rng(seed); d = np.empty(n)
    for k in range(n):
        s = rng.integers(0, N, N); d[k] = a[s].mean() - b[s].mean()
    return float(a.mean() - b.mean()), (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))

def boot_acc(correct, n=BOOT_N, seed=1):
    c = np.asarray(correct, float); N = len(c); rng = np.random.default_rng(seed)
    a = np.array([c[rng.integers(0, N, N)].mean() for _ in range(n)])
    return float(c.mean()), (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

def fast_logits(ds):
    return student_option_logits(student, [format_prompt(e["question"], e["options"]) for e in ds]).numpy()


# ----------------------------- budget planner -----------------------------
def calibrate_and_plan():
    print("calibrating reasoning-generation speed ...")
    sample = [(val_ds[i]["question"], val_ds[i]["options"]) for i in range(min(16, len(val_ds)))]
    t0 = time.time(); _ = batch_reason(sample); per = (time.time() - t0) / len(sample)
    print("  measured %.2f s/item (batched, bs=%d)" % (per, GEN_BS))
    train_gens = (HARD_N + EASY_N) if RUN_PHASE1 else 0
    train_sec = train_gens * per + (900 if RUN_PHASE1 else 0)          # +~15min training
    budget = (HOURS_LEFT - RESERVE_H) * 3600 - train_sec - N_VAL_TAU * per - 600
    phaseC_factor = 0.30 if RUN_PHASE1 else 0.0                        # ~30% routed re-gen w/ reflect
    n_test_aff = int(max(100, budget / (per * (1 + phaseC_factor))))
    n_test = min(N_TEST, len(test_ds), n_test_aff)
    print("  plan: val=%d  test=%d  train_rationales=%d" % (N_VAL_TAU, n_test, train_gens))
    eta = (N_VAL_TAU + n_test + train_gens) * per + train_sec - train_gens * per + phaseC_factor * n_test * per
    print("  projected gen+train time: %.0f min (reserve %.0f min)" % (eta / 60, RESERVE_H * 60))
    if n_test < N_TEST:
        print("  [budget] test capped from %d -> %d to fit %.1fh" % (N_TEST, n_test, HOURS_LEFT))
    return per, n_test

PER_ITEM, N_TEST_EFF = calibrate_and_plan()


# ----------------------------- Phase A : honest diagnostic -----------------------------
def phaseA():
    print("\n" + "=" * 70 + "\nPHASE A  honest routing diagnostic (untrained reasoning)\n" + "=" * 70)
    student.eval()
    Tv = fit_temperature(fast_logits(val_ds), [OPTIONS.index(e["answer"]) for e in val_ds])
    print("fast-path T=%.3f" % Tv)

    # ---- VAL: reason on zone, pick tau ----
    vsub = [val_ds[i] for i in range(min(N_VAL_TAU, len(val_ds)))]
    vg = np.array([OPTIONS.index(e["answer"]) for e in vsub])
    vfl = student_option_logits(student, [format_prompt(e["question"], e["options"]) for e in vsub]).numpy()
    vfp = _sm(vfl, Tv); vfast = vfp.argmax(1); vconf = vfp.max(1)
    vzone = np.where(vconf < ROUTE_MAX)[0]
    vr_pred = vfast.copy()
    if len(vzone):
        rp, _ = reason_preds([(vsub[i]["question"], vsub[i]["options"]) for i in vzone])
        vr_pred[vzone] = rp
    best_tau, best_vacc = TAU_GRID[0], -1
    for tau in TAU_GRID:
        route = vconf < tau
        acc = (np.where(route, vr_pred, vfast) == vg).mean()
        if acc > best_vacc: best_vacc, best_tau = acc, float(tau)
    print("val-selected tau=%.3f (val routed acc=%.4f)" % (best_tau, best_vacc))

    # ---- TEST: reason on zone, evaluate at val-selected tau with CIs ----
    tsub = [test_ds[i] for i in range(N_TEST_EFF)]
    tg = np.array([OPTIONS.index(e["answer"]) for e in tsub])
    tfl = student_option_logits(student, [format_prompt(e["question"], e["options"]) for e in tsub]).numpy()
    tfp = _sm(tfl, Tv); tfast = tfp.argmax(1); tconf = tfp.max(1)
    tzone = np.where(tconf < ROUTE_MAX)[0]
    print("reasoning on %d/%d test items (zone conf<%.2f) ..." % (len(tzone), len(tsub), ROUTE_MAX))
    tr_pred = tfast.copy(); tr_tok = np.zeros(len(tsub)); t0 = time.time()
    B = 64
    for s in range(0, len(tzone), B):
        if not have_time(700): print("  [budget] stop test reasoning at", s); tzone = tzone[:s]; break
        blk = tzone[s:s + B]
        rp, rt = reason_preds([(tsub[i]["question"], tsub[i]["options"]) for i in blk])
        tr_pred[blk] = rp; tr_tok[blk] = rt
        print("  reasoned %d/%d (%.1f min)" % (min(s + B, len(tzone)), len(tzone), (time.time() - t0) / 60))

    route = tconf < best_tau
    routed_pred = np.where(route, tr_pred, tfast)
    routed_corr = (routed_pred == tg).astype(int); fast_corr = (tfast == tg).astype(int)
    reason_corr = (tr_pred == tg).astype(int)
    racc, rci = boot_acc(routed_corr); facc, fci = boot_acc(fast_corr)
    diff, dci = boot_diff(routed_corr, fast_corr)
    tok_per_q = float(np.where(route, tr_tok, 0.0).mean())
    print("\n--- TEST (n=%d), tau=%.3f ---" % (len(tsub), best_tau))
    print("fast-only   acc=%.4f  CI[%.4f,%.4f]" % (facc, *fci))
    print("CALM-R3     acc=%.4f  CI[%.4f,%.4f]  | %.0f%% reasoned | %.1f tok/q"
          % (racc, *rci, 100 * route.mean(), tok_per_q))
    print("routed - fast = %+.4f  CI[%+.4f,%+.4f]  -> %s" %
          (diff, *dci, "SIGNIFICANT (>0)" if dci[0] > 0 else "within noise"))

    out = dict(T=Tv, tau=best_tau, val_routed_acc=float(best_vacc), n_test=len(tsub),
               fast_acc=facc, fast_ci=fci, routed_acc=racc, routed_ci=rci,
               diff=diff, diff_ci=dci, frac_reasoned=float(route.mean()), tok_per_q=tok_per_q,
               reason_only_acc=float(reason_corr.mean()))
    json.dump(out, open(os.path.join(R3, "phaseA.json"), "w"), indent=2)
    np.savez(os.path.join(R3, "phaseA_arrays.npz"), tconf=tconf, tfast=tfast, tr_pred=tr_pred,
             tg=tg, tr_tok=tr_tok)
    return out

phaseA_out = phaseA()


# ----------------------------- Phase B : train the reflect adapter -----------------------------
def _score_prompt(q, opts, rationale):
    return (format_prompt(q, opts) + "\n\nStep-by-step reasoning:\n" + rationale.strip() +
            "\n\nGiven the reasoning above, reply with ONLY one letter: A, B, C, or D.")

def _letter_logits_grad(prompts):
    enc = student_tok([stu_chat(p) for p in prompts], return_tensors="pt", padding=True,
                      truncation=True, max_length=cfg.max_src_len + RAT_TOK + 64).to(STUDENT_DEVICE)
    return _letter_logits(_fwd_last(student, enc.input_ids, enc.attention_mask))

def _adapter(name):
    try: student.set_adapter(name)
    except Exception as e: print("  set_adapter(%s):" % name, e)

def phaseB():
    from peft import LoraConfig
    import random
    print("\n" + "=" * 70 + "\nPHASE B  train reflect adapter (corrected reflective-correction loss)\n" + "=" * 70)
    student.eval(); _adapter("default")
    T = phaseA_out["T"]
    sP = _sm(fast_logits(seed_ds), T); sg = np.array([OPTIONS.index(e["answer"]) for e in seed_ds])
    spred = sP.argmax(1); sconf = sP.max(1); sok = spred == sg
    hard = [i for i in range(len(seed_ds)) if (not sok[i]) or sconf[i] < 0.5][:HARD_N]
    easy = [i for i in range(len(seed_ds)) if sok[i] and sconf[i] >= 0.7][:EASY_N]
    print("hard=%d easy=%d  -> generating rationales (batched)" % (len(hard), len(easy)))

    hr = batch_reason([(seed_ds[i]["question"], seed_ds[i]["options"]) for i in hard], max_new=RAT_TOK)
    er = batch_reason([(seed_ds[i]["question"], seed_ds[i]["options"]) for i in easy], max_new=RAT_TOK)
    ex = []
    for k, i in enumerate(hard):
        ex.append((_score_prompt(seed_ds[i]["question"], seed_ds[i]["options"], hr[k][0]), int(sg[i]), "hard", None))
    for k, i in enumerate(easy):
        ex.append((_score_prompt(seed_ds[i]["question"], seed_ds[i]["options"], er[k][0]), int(sg[i]),
                   "easy", sP[i].astype(np.float32)))
    random.Random(cfg.seed).shuffle(ex); print("train examples:", len(ex))

    rcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      bias="none", task_type="CAUSAL_LM", target_modules=list(cfg.lora_targets))
    try: student.add_adapter("reflect", rcfg)
    except Exception as e: print("  add_adapter:", e)
    _adapter("reflect")
    for n_, p_ in student.named_parameters():
        p_.requires_grad = ("lora" in n_ and "reflect" in n_)
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=RFT_LR); _use_cache(False); student.train()
    print("reflect trainable tensors:", len(params))
    step = 0
    for ep in range(RFT_EPOCHS):
        for b in range(0, len(ex), RFT_BS):
            if not have_time(500): print("  [budget] stop training"); break
            batch = ex[b:b + RFT_BS]
            logits = _letter_logits_grad([z[0] for z in batch]).float()
            gold = torch.tensor([z[1] for z in batch], device=logits.device)
            mh = torch.tensor([z[2] == "hard" for z in batch], device=logits.device)
            loss = logits.new_zeros(())
            if mh.any(): loss = loss + F.cross_entropy(logits[mh], gold[mh])
            me = ~mh
            if me.any():
                soft = torch.tensor(np.stack([z[3] for z in batch if z[2] == "easy"]),
                                    device=logits.device).float()
                loss = loss + RFT_MU * F.kl_div(F.log_softmax(logits[me], -1), soft, reduction="batchmean")
            (loss / RFT_ACC).backward(); step += 1
            if step % RFT_ACC == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip); opt.step(); opt.zero_grad()
        print("  epoch %d loss=%.4f" % (ep + 1, float(loss)))
    student.eval(); _use_cache(False)
    student.save_pretrained(os.path.join(R3, "reflect_adapter"))
    print("saved reflect adapter")


# ----------------------------- Phase C : routed eval with trained reflect -----------------------------
def phaseC():
    print("\n" + "=" * 70 + "\nPHASE C  routed evaluation with TRAINED reflect adapter\n" + "=" * 70)
    d = np.load(os.path.join(R3, "phaseA_arrays.npz"))
    tconf, tfast, tg = d["tconf"], d["tfast"], d["tg"]
    tau = phaseA_out["tau"]; route = np.where(tconf < tau)[0]
    tsub = [test_ds[i] for i in range(len(tg))]
    pred = tfast.copy(); tok = np.zeros(len(tg)); t0 = time.time(); B = 64
    _adapter("reflect")
    for s in range(0, len(route), B):
        if not have_time(400): print("  [budget] stop at", s); route = route[:s]; break
        blk = route[s:s + B]
        rp, rt = reason_preds([(tsub[i]["question"], tsub[i]["options"]) for i in blk])
        pred[blk] = rp; tok[blk] = rt
        print("  reflect-reasoned %d/%d (%.1f min)" % (min(s + B, len(route)), len(route), (time.time() - t0) / 60))
    _adapter("default")
    rc = (pred == tg).astype(int); fc = (tfast == tg).astype(int)
    racc, rci = boot_acc(rc); diff, dci = boot_diff(rc, fc)
    res = dict(tau=float(tau), routed_acc=racc, routed_ci=rci, fast_acc=float(fc.mean()),
               diff_vs_fast=diff, diff_ci=dci, frac_reasoned=float(len(route) / len(tg)),
               tok_per_q=float(np.where(np.isin(np.arange(len(tg)), route), tok, 0).mean()),
               vs_untrained_reason=racc - phaseA_out["routed_acc"])
    json.dump(res, open(os.path.join(R3, "phaseC.json"), "w"), indent=2)
    print("\nCALM-R3 (trained reflect): acc=%.4f CI[%.4f,%.4f] | routed-fast=%+.4f CI[%+.4f,%+.4f] -> %s"
          % (racc, *rci, diff, *dci, "SIGNIFICANT" if dci[0] > 0 else "within noise"))
    print("vs untrained-reasoning routed acc: %+.4f" % res["vs_untrained_reason"])
    return res

if RUN_PHASE1:
    phaseB(); phaseC_out = phaseC()
else:
    print("Phase B/C skipped (RUN_PHASE1=False).")


# ----------------------------- zip -----------------------------
import shutil
zb = os.path.join(cfg.out_dir, "calm_r3_final_artifacts")
shutil.make_archive(zb, "zip", R3)
print("\nzipped ->", zb + ".zip", "| contents:", sorted(os.listdir(R3)))
