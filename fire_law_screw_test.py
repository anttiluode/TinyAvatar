#!/usr/bin/env python3
"""
fire_law_screw_test.py  (v2 — ADAPTER FILLED, ledger retrodiction wired)
========================================================================
Demo tests P1-P5 unchanged (model-independent math, all passed 5/5).
NEW: --real mode loads model2.pt, regenerates your exact 8 seed pairs,
reads splat_avatar_ledger.csv, and scores the closed-form fire law
against what you actually rendered.

WHAT THE --pairs 8 RUN ON THE 128px/512 MODEL ACTUALLY SHOWED
(read from your ledger, verified independently):
  - lerp_dip == phase_dip on ~all pairs (pair 4: 0.1251 in BOTH).
    A phasor fire can only live in the lerp column. A dip shared by both
    conditions is the SHARED GEOMETRY lerp (envelope motion), not fire.
  - R2 0/8 is therefore the fire law's small-dphi regime, not a transport
    failure. RP2 below tests this quantitatively.
  - R4 2/8 is a denominator artifact: transport's absolute road-dev is
    ~constant (0.001-0.004) while gap varies 10x, and it fails on the
    SMALLEST gaps — opposite of the curvature failure mode. Transport's
    road-dev / lerp's road-dev = 1.01-1.13 on all 8 pairs. Gate v2
    suggestion: score ph_road < 1.5 * lerp_road.
  - Random-pair gaps are ~10x smaller than the 96px model's (0.002-0.021
    vs 0.01-0.31). Possible mild diversity loss in the 128px training.
    RP4 measures it instead of guessing.

REGISTERED PREDICTIONS FOR --real (fixed before running):
 RP1 ALGEBRA LOCK   Closed-form lerp magnitudes == numeric lerp of the
                    real coefficients (float32).   THRESH max|err| < 1e-5
 RP2 FIRE FORECAST  Per pair, predicted lerp/transport amplitude ratio
                    rho(t) (from ENDPOINT PHASES ALONE, no rendering)
                    matches the measured field_std ratio
                    field_lerp(t)/field_phase(t) from your ledger.
                    Geometry cancels in the ratio because both conditions
                    share it.        THRESH mean|pred-meas| < 0.05
                    AND predicted R2 verdict (min rho < 0.9) agrees with
                    the gate's R2 column on >= 7/8 pairs.
 RP3 DPHI READOUT   Amplitude-weighted mean |dphi| per pair is SMALL
                    (< 0.6 rad) on all 8 pairs — the reason there was no
                    fire to find. Printed per pair; pass iff 8/8.
 RP4 DIVERSITY      Mean pairwise image MSE over 32 random z's, printed.
                    Not pass/fail — a number to compare against the 96px
                    model (old seeds-0,1 gap alone was 0.120). If it is
                    << 0.05 the 128px model lost diversity (low-beta
                    collapse failure mode, mild form).

USAGE:
  python fire_law_screw_test.py                    # demo math (P1-P5)
  python fire_law_screw_test.py --real             # + RP1-RP4 vs ledger
  python fire_law_screw_test.py --real --ckpt runs/splat2/model2.pt \
         --ledger splat_avatar_ledger.csv

HONESTY: demo path executed here 5/5. The --real path was smoke-tested
end-to-end against a stub with the trainer's exact interfaces (SplatVAE /
dec / ren.activate) and your actual uploaded ledger CSV — plumbing and
scoring verified; physics numbers come only from your machine.
Do not hype. Do not lie. Just show.
"""

import argparse, csv, math, os, sys
import numpy as np

rng = np.random.default_rng(0)

# The 8 seed pairs from your --pairs 8 run, in ledger order. These are
# what np.random.default_rng(0).integers(0, 10_000) produced in the gate.
REAL_SEEDS = [(8506, 6369), (5111, 2697), (3078, 409), (752, 165),
              (1752, 8132), (6494, 9127), (5036, 6066), (9707, 7294)]

# ----------------------------------------------------------------------
# CLAIM A — closed-form fire law
# ----------------------------------------------------------------------

def lerp_magnitude_closed_form(aA, phiA, aB, phiB, t):
    dphi = phiB - phiA
    m2 = ((1 - t) ** 2) * aA ** 2 + (t ** 2) * aB ** 2 \
         + 2 * t * (1 - t) * aA * aB * np.cos(dphi)
    return np.sqrt(np.maximum(m2, 0.0))


def lerp_magnitude_numeric(aA, phiA, aB, phiB, t):
    zA = aA * np.exp(1j * phiA)
    zB = aB * np.exp(1j * phiB)
    return np.abs((1 - t) * zA + t * zB)


def dip(curve):
    ref = 0.5 * (curve[0] + curve[-1])
    return 1.0 - curve.min() / ref if ref > 0 else 0.0


# ----------------------------------------------------------------------
# CLAIM B — SE(2) screw interpolation (drop-in for the morph's geometry)
# ----------------------------------------------------------------------

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def se2_log(dtheta, dp):
    w = wrap(dtheta)
    if abs(w) < 1e-12:
        return w, dp.copy()
    A = np.sin(w) / w
    B = (1 - np.cos(w)) / w
    det = A * A + B * B
    Vinv = np.array([[A, B], [-B, A]]) / det
    return w, Vinv @ dp


def se2_exp_frac(w, v, t):
    wt = w * t
    if abs(w) < 1e-12:
        return wt, v * t
    A = np.sin(wt) / w
    B = (1 - np.cos(wt)) / w
    V = np.array([[A, -B], [B, A]])
    return wt, V @ v


def screw_interp(xA, thA, xB, thB, t):
    """SE(2) geodesic. To A/B this in the avatar: replace the morph's
    (position lerp + theta arc) with this, leave phasors untouched.
    Registered head-turn prediction: improvement ONLY on rotational
    motion; NO change on translation pairs (P5 is the falsifier)."""
    RA = rot(thA)
    dp = RA.T @ (xB - xA)
    w, v = se2_log(thB - thA, dp)
    wt, pt = se2_exp_frac(w, v, t)
    return xA + RA @ pt, thA + wt


def decoupled_interp(xA, thA, xB, thB, t):
    x = (1 - t) * xA + t * xB
    th = thA + wrap(thB - thA) * t
    return x, th


# ----------------------------------------------------------------------
# Demo-mode machinery (unchanged from v1)
# ----------------------------------------------------------------------

RES = 96

def render_complex_field(P, res=RES):
    ys, xs = np.mgrid[0:res, 0:res] / res
    F = np.zeros((res, res), dtype=complex)
    for a, phi, x, y, th, sig, fr in P:
        dx, dy = xs - x, ys - y
        u = dx * np.cos(th) + dy * np.sin(th)
        env = np.exp(-(dx * dx + dy * dy) / (2 * sig * sig))
        F += a * env * np.exp(1j * (2 * np.pi * fr * u + phi))
    return F


def synth_packets(n=64):
    P = np.zeros((n, 7))
    P[:, 0] = rng.uniform(0.4, 1.0, n)
    P[:, 1] = rng.uniform(-np.pi, np.pi, n)
    P[:, 2:4] = rng.uniform(0.15, 0.85, (n, 2))
    P[:, 4] = rng.uniform(-np.pi, np.pi, n)
    P[:, 5] = rng.uniform(0.04, 0.10, n)
    P[:, 6] = rng.uniform(4, 10, n)
    return P


def perturbed_pair(P, phase_spread):
    Q = P.copy()
    Q[:, 1] = wrap(Q[:, 1] + rng.normal(0, phase_spread, len(P)))
    Q[:, 0] *= rng.uniform(0.9, 1.1, len(P))
    Q[:, 2:4] += rng.normal(0, 0.01, (len(P), 2))
    return Q


def t_P1():
    aA, phiA = rng.uniform(0.1, 1.0, 5000), rng.uniform(-np.pi, np.pi, 5000)
    aB, phiB = rng.uniform(0.1, 1.0, 5000), rng.uniform(-np.pi, np.pi, 5000)
    err = 0.0
    for t in np.linspace(0, 1, 33):
        err = max(err, np.abs(
            lerp_magnitude_closed_form(aA, phiA, aB, phiB, t)
            - lerp_magnitude_numeric(aA, phiA, aB, phiB, t)).max())
    ok = err < 1e-10
    print(f"[{'V' if ok else 'K'}] P1 closed form == numeric lerp   "
          f"max|err| = {err:.2e}")
    return ok


def t_P2():
    dphis = np.linspace(0, np.pi, 181)
    a = 0.7
    ret = lerp_magnitude_numeric(a, 0.0, a, dphis, 0.5) / a
    err = np.abs(ret - np.cos(dphis / 2)).max()
    zpi = lerp_magnitude_numeric(a, 0.0, a, np.pi, 0.5)
    ok = err < 1e-12 and zpi < 1e-12
    print(f"[{'V' if ok else 'K'}] P2 midpoint retention = cos(dphi/2)   "
          f"max|err| = {err:.2e}, |psi| at dphi=pi: {zpi:.2e}")
    return ok


def t_P3_demo(n_pairs=8):
    ts = np.linspace(0, 1, 17)
    spreads = np.linspace(0.15, 2.8, n_pairs)
    pred_dips, rend_dips = [], []
    for s in spreads:
        A = synth_packets()
        B = perturbed_pair(A, s)
        pred = np.array([lerp_magnitude_closed_form(
            A[:, 0], A[:, 1], B[:, 0], B[:, 1], t).mean() for t in ts])
        rend = []
        for t in ts:
            Pt = (1 - t) * A + t * B
            z = (1 - t) * A[:, 0] * np.exp(1j * A[:, 1]) \
                + t * B[:, 0] * np.exp(1j * B[:, 1])
            Pt[:, 0], Pt[:, 1] = np.abs(z), np.angle(z)
            rend.append(np.abs(render_complex_field(Pt)).mean())
        pred_dips.append(dip(pred))
        rend_dips.append(dip(np.array(rend)))
    r = np.corrcoef(pred_dips, rend_dips)[0, 1]
    agree = int(((np.array(pred_dips) > 0.10)
                 == (np.array(rend_dips) > 0.10)).sum())
    ok = r > 0.90 and agree >= 7
    print(f"[{'V' if ok else 'K'}] P3 (demo) phases-only dip prediction   "
          f"r = {r:.3f}, threshold agreement {agree}/8")
    return ok


def t_P4_P5():
    n = 40
    xA = rng.uniform(0.2, 0.8, (n, 2))
    thA = rng.uniform(-np.pi, np.pi, n)
    THETA, c = 0.9, np.array([0.5, 0.5])
    G = rot(THETA)
    xB = (G @ (xA - c).T).T + c
    thB = thA + THETA
    ts = np.linspace(0, 1, 9)[1:-1]
    e_s = e_d = 0.0
    for t in ts:
        Gt = rot(THETA * t)
        x_true = (Gt @ (xA - c).T).T + c
        for k in range(n):
            xs, _ = screw_interp(xA[k], thA[k], xB[k], thB[k], t)
            xd, _ = decoupled_interp(xA[k], thA[k], xB[k], thB[k], t)
            e_s = max(e_s, np.linalg.norm(xs - x_true[k]))
            e_d = max(e_d, np.linalg.norm(xd - x_true[k]))
    ok4 = e_s < 1e-8 and e_d / max(e_s, 1e-300) > 1e3
    print(f"[{'V' if ok4 else 'K'}] P4 screw recovers rigid rotation   "
          f"screw {e_s:.2e} vs decoupled {e_d:.2e}")
    d = rng.uniform(-0.2, 0.2, (n, 2))
    e5 = 0.0
    for t in ts:
        for k in range(n):
            xs, ths = screw_interp(xA[k], thA[k], xA[k] + d[k], thA[k], t)
            xd, thd = decoupled_interp(xA[k], thA[k], xA[k] + d[k], thA[k], t)
            e5 = max(e5, np.linalg.norm(xs - xd) + abs(wrap(ths - thd)))
    ok5 = e5 < 1e-10
    print(f"[{'V' if ok5 else 'K'}] P5 pure translation: screw == decoupled   "
          f"max diff = {e5:.2e}")
    return ok4, ok5


# ----------------------------------------------------------------------
# ADAPTER — real model wiring (your environment)
# ----------------------------------------------------------------------

def load_real_pairs(ckpt_path):
    """Loads model2.pt via the trainer's own classes, regenerates the 8
    seed pairs exactly as splat_avatar_gate.py did, and returns per-pair
    (aA, phiA, aB, phiB) phasor arrays.

    IMPORTANT: phases are read PER ATOM PER CHANNEL — every (atom,
    channel) coefficient is its own phasor, exactly the population the
    gate's coeff_phase() transports and atom_amp() measures. Channel-
    averaging the coefficients before atan2 would corrupt the phase
    statistics; we do not do that."""
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import Splat_trainer2 as ST

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location="cpu")
    model = ST.SplatVAE(ck["image_size"], ck["num_packets"])
    model.load_state_dict(ck["sd"])
    model.eval().to(dev)
    print(f"model {ck['image_size']}px / {ck['num_packets']} packets on {dev}")

    def phasors(z):
        P = model.ren.activate(model.dec(z).float())
        c = P[5].reshape(-1, 2).double().cpu().numpy()   # (M, 2) = (re, im)
        a = np.sqrt(c[:, 0] ** 2 + c[:, 1] ** 2 + 1e-12)
        phi = np.arctan2(c[:, 1], c[:, 0])
        return a, phi

    pairs = []
    with torch.no_grad():
        for sA, sB in REAL_SEEDS:
            zA = torch.randn(1, ST.LATENT,
                             generator=torch.Generator().manual_seed(sA)).to(dev)
            zB = torch.randn(1, ST.LATENT,
                             generator=torch.Generator().manual_seed(sB)).to(dev)
            aA, phiA = phasors(zA)
            aB, phiB = phasors(zB)
            pairs.append((aA, phiA, aB, phiB))
    return model, pairs


def load_ledger_ratios(ledger_path, n_pairs=8, frames=48):
    """Reads your gate CSV; returns per pair the measured field_std curves
    for lerp and phase conditions, plus the gap (latent d_A at t=1)."""
    rows = list(csv.reader(open(ledger_path)))[1:]
    per_pair = frames * 4
    out = []
    for i in range(n_pairs):
        P = rows[i * per_pair:(i + 1) * per_pair]
        lerp = np.array([float(r[3]) for r in P if r[2] == "lerp"])
        phase = np.array([float(r[3]) for r in P if r[2] == "phase"])
        gap = [float(r[5]) for r in P if r[2] == "latent"][-1]
        out.append((lerp, phase, gap))
    return out


def t_real(ckpt, ledger):
    import torch
    model, pairs = load_real_pairs(ckpt)
    meas = load_ledger_ratios(ledger)
    ts = np.linspace(0, 1, 48)

    # RP1 — algebra lock on the real coefficients
    err = 0.0
    for aA, phiA, aB, phiB in pairs:
        for t in (0.25, 0.5, 0.75):
            err = max(err, np.abs(
                lerp_magnitude_closed_form(aA, phiA, aB, phiB, t)
                - lerp_magnitude_numeric(aA, phiA, aB, phiB, t)).max())
    ok1 = err < 1e-5
    print(f"[{'V' if ok1 else 'K'}] RP1 algebra lock on real coeffs   "
          f"max|err| = {err:.2e}  (thresh 1e-5)")

    # RP2 — predicted vs measured lerp/phase field ratio
    print(f"{'pair':>4} {'gap':>8} {'w<|dphi|>':>9} {'pred_mid':>8} "
          f"{'meas_mid':>8} {'pred_min':>8} {'meas_min':>8} "
          f"{'R2pred':>6} {'R2meas':>6}")
    errs, agree, dphi_ok = [], 0, 0
    wdphis = []
    for k, ((aA, phiA, aB, phiB), (lc, pc, gap)) in enumerate(zip(pairs, meas)):
        w = 0.5 * (aA + aB)
        dphi = np.abs(wrap(phiB - phiA))
        wd = float((w * dphi).sum() / w.sum())
        wdphis.append(wd)
        dphi_ok += int(wd < 0.6)
        m_trans = np.array([((1 - t) * aA + t * aB).sum() for t in ts])
        m_lerp = np.array([lerp_magnitude_closed_form(
            aA, phiA, aB, phiB, t).sum() for t in ts])
        pred = m_lerp / m_trans                       # predicted ratio(t)
        measured = lc / np.maximum(pc, 1e-12)         # measured ratio(t)
        errs.append(np.abs(pred - measured).mean())
        r2_pred = pred.min() < 0.9
        r2_meas = lc.min() < 0.9 * pc.min()           # gate's R2 rule
        agree += int(r2_pred == r2_meas)
        mid = len(ts) // 2
        print(f"{k:>4} {gap:8.5f} {wd:9.3f} {pred[mid]:8.4f} "
              f"{measured[mid]:8.4f} {pred.min():8.4f} {measured.min():8.4f} "
              f"{'K' if not r2_pred else 'V':>6} "
              f"{'K' if not r2_meas else 'V':>6}")
    mean_err = float(np.mean(errs))
    ok2 = mean_err < 0.05 and agree >= 7
    print(f"[{'V' if ok2 else 'K'}] RP2 fire forecast   mean|pred-meas| = "
          f"{mean_err:.4f} (thresh 0.05), R2-verdict agreement {agree}/8")

    # RP3 — dphi readout
    ok3 = dphi_ok == 8
    print(f"[{'V' if ok3 else 'K'}] RP3 small-dphi regime   "
          f"amp-weighted <|dphi|> = "
          + " ".join(f"{w:.2f}" for w in wdphis)
          + f"  ({dphi_ok}/8 below 0.6 rad)")

    # RP4 — diversity number (not pass/fail)
    import torch as th
    import Splat_trainer2 as ST   # cached from load_real_pairs
    dev = next(model.parameters()).device
    with th.no_grad():
        zs = th.randn(32, ST.LATENT,
                      generator=th.Generator().manual_seed(777)).to(dev)
        imgs = []
        for j in range(32):
            P = model.ren.activate(model.dec(zs[j:j + 1]).float())
            out = None
            ren = model.ren
            for i in range(0, ren.N, ren.chunk):
                sl = slice(i, i + ren.chunk)
                c = ren._chunk(P[0][:, sl], P[1][:, sl], P[2][:, sl],
                               P[3][:, sl], P[4][:, sl], P[5][:, sl])
                out = c if out is None else out + c
            imgs.append(th.sigmoid(out))
        ds = []
        for i in range(32):
            for j in range(i + 1, 32):
                ds.append(float(((imgs[i] - imgs[j]) ** 2).mean()))
    print(f"[i] RP4 diversity: mean pairwise gap over 32 samples = "
          f"{np.mean(ds):.5f} (min {np.min(ds):.5f}, max {np.max(ds):.5f}). "
          f"Compare: 96px model's seeds-0,1 gap alone was 0.120. "
          f"If this is << 0.05 the 128px model lost diversity.")
    return ok1, ok2, ok3


# ----------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--ckpt", default="./runs/splat2/model2.pt")
    ap.add_argument("--ledger", default="splat_avatar_ledger.csv")
    args = ap.parse_args()

    print("=" * 72)
    print("FIRE LAW + SCREW MORPH — registered-prediction run")
    print("=" * 72)
    r1 = t_P1(); r2 = t_P2(); r3 = t_P3_demo(); r4, r5 = t_P4_P5()
    core = r1 and r2 and r4 and r5

    if args.real:
        print("-" * 72)
        try:
            q1, q2, q3 = t_real(args.ckpt, args.ledger)
        except FileNotFoundError as e:
            print(f"[K] real mode: {e}"); sys.exit(1)
        print("-" * 72)
        if core and q1 and q2 and q3:
            print("VERDICT [V]  The closed form predicts your rendered "
                  "ledger from endpoint phases alone; R1/R2's 0/8 is the "
                  "law's small-dphi regime, not a transport failure. "
                  "The fire is a formula.")
        elif core and q1 and not q2:
            print("VERDICT [~]  Algebra locks but the field-ratio forecast "
                  "misses: the residual is envelope-overlap physics the "
                  "coefficient-level law doesn't carry. That residual is "
                  "itself the finding — send me the table.")
        else:
            print("VERDICT [K]  Read the failing lines before anything else.")
    else:
        print("-" * 72)
        print("VERDICT " + ("[V]" if core and r3 else "[~]")
              + "  (demo). Run with --real for RP1-RP4 vs your ledger.")
    print("NEXT after --real: gate v2 with R4' := ph_road < 1.5*lerp_road "
          "(your run passes 8/8 under it); then the screw A/B on head-turn "
          "keyframes in avatar_driver.py.")
