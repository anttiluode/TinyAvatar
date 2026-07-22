#!/usr/bin/env python3
# ======================================================================
# splat_pulse.py — live "pulse" tracker for the Gabor-splat medium.
#
# Drop next to tiny_avatar3.py + splat_trainer3v2.py + a trained .pt.
# Reuses the avatar driver's own machinery by import (FaceFramer,
# pursue, render_image, trainer loader); falls back to local copies if
# tiny_avatar3 / PyQt6 aren't importable, so it also runs standalone.
#
# WHAT IT SHOWS, per display frame, while the avatar is being driven
# (webcam keyframes or a latent walk) through the same pursue() loop
# the app uses:
#
#   [A] avatar render with a packet "quiver": each stable packet drawn
#       as an oriented tick at (p_k), length ~ amplitude, color = its
#       instantaneous phase velocity dPhi_k (blue<-0->red). Motion
#       shows up as color waves sweeping the face — the pulse.
#   [B] pulse map: packets in (px,py) space, color = dPhi_k, radius ~
#       amplitude; the 3 tracked orbit packets ringed white.
#   [C] instruments:
#         - scrolling Phi traces of the 3 tracked packets (the live
#           Takens signal; this is phase_orbit.py's orbit, streaming)
#         - band meters: LOW f<3 / MID 3-8 / HIGH f>=8 residual
#           coherence  R_band = |sum m e^{i e_k}| / sum m,  where
#           e_k = dPhi_k − (−2π f_k u_k·v_k) is the residual after
#           removing the keyframe-implied dispersion prediction
#           (v_k = alpha (p_T − p), same quantity the pursuit uses).
#           R -> 1: packets move as the dispersion form predicts.
#           R -> 0: phases scatter — the fire-side of the medium.
#
# HONESTY NOTES (do not hype):
#   - The band-coherence meter is an INSTRUMENT built on the dispersion
#     FORM. The dispersion law itself is chain-certified (selftest
#     r=+0.915) but still unmeasured on real pose video (P2 pending),
#     so read R as "agreement with the predicted form", not as a
#     certified physical invariant.
#   - The "fire = decoherence, hallucination-regularizer" story from
#     the framework doc is a hypothesis. This tool measures the
#     quantities that story is about; it does not confirm the story.
#     The registered, checkable claim it enables:
#       [PULSE-1] during smooth in-manifold driving, R_LOW stays above
#                 R_HIGH almost always (low-f skeleton holds lock while
#                 high-f detail scatters first).
#       [PULSE-2] framing breaks / off-manifold input produce a
#                 simultaneous collapse of R across bands, preceding
#                 visible recon degradation.
#     Both are logged to pulse_log.csv so they can be scored after a
#     session instead of eyeballed.
#   - No coherence *dampener* is applied anywhere in this tool. Build
#     the intervention only after the measurement says where the lever
#     is.
#
# USAGE
#   python splat_pulse.py --walk                    # latent walk, live window
#   python splat_pulse.py --cam 0                   # webcam-driven
#   python splat_pulse.py --walk --record out.mp4 --nframes 300   # headless
#   keys: q quit   m cycle pursuit mode   space pause
# ======================================================================

import argparse, csv, glob, math, os, sys, time
from collections import deque
import numpy as np

# ---------------------------------------------------------------------
# borrow the avatar app's machinery; fall back to local copies
# ---------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TA = None
try:
    import tiny_avatar3 as TA
except Exception as e:
    print(f"(tiny_avatar3 not importable: {type(e).__name__}: {e} — "
          "using built-in fallbacks)")

if TA is not None:
    FaceFramer   = TA.FaceFramer
    pursue       = TA.pursue
    clone_params = TA.clone_params
    render_image = TA.render_image
    normalize_crop = TA.normalize_crop
    find_trainer = TA.find_trainer
    import_trainer = TA.import_trainer
else:
    # ---- minimal fallbacks (verbatim math from tiny_avatar3) ----
    def find_trainer():
        for pat in ("splat_trainer3*.py", "splat_trainer*.py"):
            hits = sorted(glob.glob(os.path.join(HERE, pat)))
            if hits:
                return hits[-1]
        raise RuntimeError("no splat_trainer*.py next to splat_pulse.py")

    def import_trainer(path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("splat_trainer", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def normalize_crop(x, tgt_mean=0.52, tgt_std=0.26):
        m, s = float(x.mean()), float(x.std()) + 1e-6
        return np.clip((x - m) / s * tgt_std + tgt_mean, 0.0, 1.0)

    def render_image(ren, P):
        import torch
        px, py, s, th, f, c = P
        out = None
        for i in range(0, ren.N, ren.chunk):
            sl = slice(i, i + ren.chunk)
            cc = ren._chunk(px[:, sl], py[:, sl], s[:, sl],
                            th[:, sl], f[:, sl], c[:, sl])
            out = cc if out is None else out + cc
        return torch.sigmoid(out)

    def clone_params(P):
        return tuple(t.clone() for t in P)

    def _arc_step(a, b, alpha):
        import torch
        d = torch.remainder(b - a + math.pi, 2 * math.pi) - math.pi
        return a + alpha * d

    def pursue(P, T, alpha, mode):
        import torch
        px, py, s, th, f, c = P
        pxT, pyT, sT, thT, fT, cT = T
        L = lambda a, b: a + alpha * (b - a)
        s2, f2 = L(s, sT), L(f, fT)
        px2, py2 = L(px, pxT), L(py, pyT)
        th2 = _arc_step(th, thT, alpha)
        if mode == "lerp":
            c2 = L(c, cT)
        else:                                   # phase transport
            a_, b_ = c[..., 0], c[..., 1]
            aT, bT = cT[..., 0], cT[..., 1]
            m = torch.sqrt(a_ * a_ + b_ * b_ + 1e-12)
            mT = torch.sqrt(aT * aT + bT * bT + 1e-12)
            ph = torch.atan2(b_, a_); phT = torch.atan2(bT, aT)
            m2 = L(m, mT)
            ph2 = _arc_step(ph, phT, alpha)
            c2 = torch.stack([m2 * torch.cos(ph2),
                              m2 * torch.sin(ph2)], dim=-1)
        return (px2, py2, s2, th2, f2, c2)

    class FaceFramer:
        def __init__(self, margin=0.35, ema=0.30, every=2):
            import cv2 as cv
            cpath = os.path.join(cv.data.haarcascades,
                                 "haarcascade_frontalface_default.xml")
            self.det = cv.CascadeClassifier(cpath)
            if self.det.empty():
                self.det = None
            self.margin, self.ema, self.every = margin, ema, every
            self.box, self.f = None, 0
        def crop(self, fr):
            import cv2 as cv
            H, W = fr.shape[:2]
            if self.det is not None and self.f % self.every == 0:
                g = cv.cvtColor(fr, cv.COLOR_BGR2GRAY)
                det = self.det.detectMultiScale(g, 1.15, 5, minSize=(80, 80))
                if len(det):
                    x, y, w, h = max(det, key=lambda b: b[2] * b[3])
                    m = self.margin * max(w, h)
                    cx, cy = x + w / 2, y + h / 2
                    half = max(w, h) / 2 + m
                    if self.box is None:
                        self.box = (cx, cy, half)
                    else:
                        a = self.ema
                        self.box = (a * cx + (1 - a) * self.box[0],
                                    a * cy + (1 - a) * self.box[1],
                                    a * half + (1 - a) * self.box[2])
            self.f += 1
            if self.box is None:
                s = min(H, W)
                return fr[(H - s)//2:(H + s)//2, (W - s)//2:(W + s)//2]
            cx, cy, half = self.box
            s = int(half)
            x0, x1 = int(max(cx - s, 0)), int(min(cx + s, W))
            y0, y1 = int(max(cy - s, 0)), int(min(cy + s, H))
            c = fr[y0:y1, x0:x1]
            if c.size:
                return c
            s = min(H, W)
            return fr[(H - s)//2:(H + s)//2, (W - s)//2:(W + s)//2]

# ---------------------------------------------------------------------
# pulse math
# ---------------------------------------------------------------------
BANDS = (("LOW",  0.0, 3.0), ("MID", 3.0, 8.0), ("HIGH", 8.0, 99.0))

class PulseTracker:
    """Streams per-packet phasors out of the live pursued state P and
    keeps the composite phase Phi_k (referenced at an EMA of each
    packet's center — same construction phase_orbit.py certified)."""

    def __init__(self, npk):
        self.N = npk
        self.xb = None            # EMA centers (2,N)
        self.Zp = None            # previous composite phasor (N,)
        self.Phi = np.zeros(npk)  # cumulative unwrapped phase
        self.picks = None

    def step(self, P, T, alpha):
        px, py, sg, th, fq, co = [t[0].detach().cpu().numpy() for t in P]
        z = (co[..., 0] + 1j * co[..., 1]).mean(-1)          # (N,) lum phasor
        ctr = np.stack([px, py])
        self.xb = ctr if self.xb is None else 0.98 * self.xb + 0.02 * ctr
        carrier = 2 * np.pi * fq * (np.cos(th) * (self.xb[0] - px)
                                    + np.sin(th) * (self.xb[1] - py))
        Z = z * np.exp(1j * carrier)
        if self.Zp is None:
            dphi = np.zeros(self.N)
        else:
            dphi = np.angle(Z * np.conj(self.Zp))
        self.Zp = Z
        self.Phi += dphi
        amp = np.abs(Z)

        # keyframe-implied per-packet velocity + dispersion-form residual
        pxT, pyT = [t[0].detach().cpu().numpy() for t in T[:2]]
        vx, vy = alpha * (pxT - px), alpha * (pyT - py)
        pred = -2 * np.pi * fq * (np.cos(th) * vx + np.sin(th) * vy)
        resid = np.angle(np.exp(1j * (dphi - pred)))

        # per-band residual coherence R = |sum m e^{i e}| / sum m
        R = {}
        for name, lo, hi in BANDS:
            m = (fq >= lo) & (fq < hi) & (amp > 0.02)
            if m.sum() < 3:
                R[name] = np.nan
            else:
                R[name] = float(np.abs((amp[m] * np.exp(1j * resid[m])).sum())
                                / (amp[m].sum() + 1e-12))
        if self.picks is None:
            order = np.argsort(-amp)
            picks = [int(order[0])]
            for k in order[1:]:
                if all(np.hypot(px[k] - px[j], py[k] - py[j]) > 0.15
                       for j in picks):
                    picks.append(int(k))
                if len(picks) == 3:
                    break
            while len(picks) < 3:
                picks.append(int(order[len(picks)]))
            self.picks = picks
        return dict(px=px, py=py, th=th, fq=fq, amp=amp, dphi=dphi,
                    Phi=self.Phi.copy(), R=R, picks=self.picks)

# ---------------------------------------------------------------------
# drawing (pure cv2, no matplotlib in the loop)
# ---------------------------------------------------------------------
PANE = 384
TRAIL = 240

def phase_color(d, scale=0.25):
    """dphi -> BGR: blue negative, gray zero, red positive."""
    t = float(np.clip(d / scale, -1, 1))
    if t >= 0:
        return (60, int(60 + 40 * (1 - t)), int(100 + 155 * t))
    t = -t
    return (int(100 + 155 * t), int(60 + 40 * (1 - t)), 60)

def draw_panels(cv, av_rgb, S, hist, mode, kf, alpha, fps):
    H = PANE
    # [A] avatar + quiver
    A = cv.resize(av_rgb[:, :, ::-1], (H, H),
                  interpolation=cv.INTER_NEAREST).copy()
    amax = S["amp"].max() + 1e-9
    for k in range(len(S["px"])):
        if S["amp"][k] < 0.02:
            continue
        x, y = S["px"][k] * H, S["py"][k] * H
        L = 4 + 10 * S["amp"][k] / amax
        dx, dy = math.cos(S["th"][k]) * L, math.sin(S["th"][k]) * L
        cv.line(A, (int(x - dx), int(y - dy)), (int(x + dx), int(y + dy)),
                phase_color(S["dphi"][k]), 1, cv.LINE_AA)
    cv.putText(A, "avatar + pulse quiver", (8, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv.LINE_AA)

    # [B] pulse map
    B = np.full((H, H, 3), 18, np.uint8)
    for k in range(len(S["px"])):
        if S["amp"][k] < 0.02:
            continue
        x, y = int(S["px"][k] * H), int(S["py"][k] * H)
        r = int(2 + 6 * S["amp"][k] / amax)
        cv.circle(B, (x, y), r, phase_color(S["dphi"][k]), -1, cv.LINE_AA)
    for j, k in enumerate(S["picks"]):
        x, y = int(S["px"][k] * H), int(S["py"][k] * H)
        cv.circle(B, (x, y), 11, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(B, f"k{k}", (x + 12, y + 4), cv.FONT_HERSHEY_SIMPLEX,
                   0.4, (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(B, "pulse map  (dPhi: blue - / red +)", (8, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv.LINE_AA)

    # [C] instruments
    C = np.full((H, H, 3), 12, np.uint8)
    # scrolling Phi traces (top 2/3)
    h1 = int(H * 0.62)
    cv.putText(C, "Phi traces (tracked packets)", (8, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv.LINE_AA)
    if len(hist) > 2:
        arr = np.array([[h["Phi"][k] for k in S["picks"]] for h in hist])
        arr = arr - arr[-1]                       # anchor now at 0
        lim = max(1.0, float(np.abs(arr).max()))
        cols = [(80, 200, 255), (120, 255, 140), (255, 170, 90)]
        n = len(arr)
        for j in range(3):
            pts = [(int(30 + (H - 40) * i / max(TRAIL - 1, 1)),
                    int(h1 / 2 - (h1 / 2 - 26) * arr[i, j] / lim))
                   for i in range(n)]
            for p, q in zip(pts[:-1], pts[1:]):
                cv.line(C, p, q, cols[j], 1, cv.LINE_AA)
            cv.putText(C, f"k{S['picks'][j]}", (H - 52, 36 + 16 * j),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, cols[j], 1, cv.LINE_AA)
        cv.putText(C, f"+/-{lim:.1f} rad", (8, h1 - 6),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1,
                   cv.LINE_AA)
    # band coherence bars (bottom)
    cv.putText(C, "band residual coherence R", (8, h1 + 18),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv.LINE_AA)
    for i, (name, _, _) in enumerate(BANDS):
        r = S["R"].get(name, np.nan)
        x0 = 20 + i * ((H - 40) // 3)
        w = (H - 40) // 3 - 16
        y0, y1 = h1 + 30, H - 34
        cv.rectangle(C, (x0, y0), (x0 + w, y1), (60, 60, 60), 1)
        if not np.isnan(r):
            hgt = int((y1 - y0) * r)
            col = (90, 220, 120) if r > 0.7 else \
                  ((90, 200, 240) if r > 0.4 else (80, 80, 250))
            cv.rectangle(C, (x0, y1 - hgt), (x0 + w, y1), col, -1)
        cv.putText(C, f"{name} {'' if np.isnan(r) else f'{r:.2f}'}",
                   (x0, H - 16), cv.FONT_HERSHEY_SIMPLEX, 0.45,
                   (220, 220, 220), 1, cv.LINE_AA)
    out = np.concatenate([A, B, C], axis=1)
    cv.putText(out, f"{mode}  kf {kf}  alpha {alpha:.2f}  {fps:4.0f} fps"
               "   [q quit  m mode  space pause]",
               (8, PANE - 6), cv.FONT_HERSHEY_SIMPLEX, 0.45,
               (200, 200, 200), 1, cv.LINE_AA)
    return out

# ---------------------------------------------------------------------
def main():
    import cv2 as cv
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="trained .pt (default: "
                    "newest model2.pt under runs/ or cwd)")
    ap.add_argument("--cam", type=int, default=None)
    ap.add_argument("--walk", action="store_true")
    ap.add_argument("--kf", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--mode", default="phase",
                    choices=["phase", "lerp"])
    ap.add_argument("--walk-step", type=float, default=2.5)
    ap.add_argument("--z-max", type=float, default=12.0)
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--no-norm", action="store_true")
    ap.add_argument("--record", default=None, help="write mp4 instead of "
                    "opening a window (headless)")
    ap.add_argument("--nframes", type=int, default=300,
                    help="frames to run in --record mode")
    ap.add_argument("--log", default="pulse_log.csv")
    args = ap.parse_args()
    if args.cam is None and not args.walk:
        args.walk = True

    ckpt = args.ckpt
    if ckpt is None:
        cand = (sorted(glob.glob(os.path.join(HERE, "runs", "*", "*.pt")))
                + sorted(glob.glob(os.path.join(HERE, "*.pt"))))
        if not cand:
            sys.exit("no .pt found — pass --ckpt")
        ckpt = cand[-1]
    ST = import_trainer(find_trainer())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = ST.SplatVAE(ck["image_size"], ck["num_packets"])
    model.load_state_dict(ck["sd"]); model.eval().to(dev)
    ren = model.ren
    LATENT = getattr(ST, "LATENT", 128)
    print(f"{os.path.basename(ckpt)}: {ck['image_size']}px / "
          f"{ck['num_packets']} packets on {dev}")

    tracker = PulseTracker(ck["num_packets"])
    hist = deque(maxlen=TRAIL)
    logf = open(args.log, "w", newline="")
    logw = csv.writer(logf)
    logw.writerow(["frame", "R_LOW", "R_MID", "R_HIGH",
                   "Phi_a", "Phi_b", "Phi_c", "mode"])

    cap = framer = None
    if args.cam is not None:
        cap = cv.VideoCapture(args.cam)
        if not cap.isOpened():
            sys.exit("webcam failed — use --walk")
        framer = None if args.no_align else FaceFramer()
    g = torch.Generator().manual_seed(0)
    z = torch.randn(1, LATENT, generator=g).to(dev)

    writer = None
    if args.record:
        writer = cv.VideoWriter(args.record,
                                cv.VideoWriter_fourcc(*"mp4v"),
                                30, (PANE * 3, PANE))

    def keyframe_from_z(zz):
        with torch.no_grad():
            return ren.activate(model.dec(zz).float())

    P = T = None
    mode, paused = args.mode, False
    f, fps, t_last = 0, 0.0, time.time()
    try:
        while True:
            if not paused:
                if cap is not None:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    crop = framer.crop(frame) if framer is not None else None
                    if crop is None:
                        h, w = frame.shape[:2]; s = min(h, w)
                        crop = frame[(h - s)//2:(h + s)//2,
                                     (w - s)//2:(w + s)//2]
                    x = cv.resize(crop, (ren.H, ren.W))[:, :, ::-1] \
                        .astype(np.float32) / 255.0
                    if not args.no_norm:
                        x = normalize_crop(x)
                    if f % max(1, args.kf) == 0 or T is None:
                        xt = torch.from_numpy(np.ascontiguousarray(
                            x.transpose(2, 0, 1)))[None].to(dev)
                        with torch.no_grad():
                            mu, _ = model.enc(xt)
                            T = keyframe_from_z(mu)
                else:
                    if f % max(1, args.kf) == 0 or T is None:
                        step = torch.randn(1, LATENT, generator=g).to(dev)
                        z2 = z + args.walk_step * step
                        z2 = z2 * min(1.0, args.z_max / (z2.norm() + 1e-9))
                        z = z2
                        T = keyframe_from_z(z)
                if P is None:
                    P = clone_params(T)
                with torch.no_grad():
                    P = pursue(P, T, args.alpha, mode)
                    img = render_image(ren, P)
                av = (img[0].clamp(0, 1) * 255).byte() \
                    .permute(1, 2, 0).cpu().numpy()
                S = tracker.step(P, T, args.alpha)
                hist.append(S)
                logw.writerow([f] + [f"{S['R'].get(n, float('nan')):.4f}"
                                     for n, _, _ in BANDS]
                              + [f"{S['Phi'][k]:.4f}" for k in S["picks"]]
                              + [mode])
                now = time.time()
                fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6)
                t_last = now
                canvas = draw_panels(cv, av, S, hist, mode,
                                     args.kf, args.alpha, fps)
                f += 1
            if writer is not None:
                writer.write(canvas)
                if f >= args.nframes:
                    break
            else:
                cv.imshow("splat pulse", canvas)
                k = cv.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("m"):
                    mode = "lerp" if mode == "phase" else "phase"
                if k == ord(" "):
                    paused = not paused
    finally:
        logf.close()
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
            print(f"wrote {args.record} ({f} frames)")
        else:
            cv.destroyAllWindows()
        print(f"log: {args.log}  ({f} frames) — score PULSE-1/2 from it")

if __name__ == "__main__":
    main()
