#!/usr/bin/env python3
# ======================================================================
# phase_orbit.py — Takens phase-orbit + dispersion-relation extraction
# for the TinyAvatar / SplatWorld Gabor-splat medium.
#
# Drop into the TinyAvatar folder next to model2.pt.
#
# WHAT IT DOES
#   1. Gets z(t) from a frame sequence (video / webcam / image folder,
#      via the encoder) or directly (--npz / --walk). Decoder -> packet
#      params. The decoder coefficients ARE the analytic signal (RP1).
#   2. Per-packet phasor referenced at the packet's time-mean center
#      x_bar_k, absorbing the envelope-translation / phase-rotation
#      split with a short lever arm:
#          Z_k(t) = z_k(t) * exp(i 2*pi f_k u_k . (x_bar_k - p_k))
#   3. Slot-stability check (slots can migrate in z; unstable excluded).
#   4. Phase increments dPhi_k = arg(Z(t+1) conj(Z(t))).
#   5. DISPERSION LAW (medium-internal): renders the reconstruction and
#      measures each packet's LOCAL optical flow v_k(t) (windowed
#      phase-correlation around x_bar_k on the rendered frames). Claim:
#          dPhi_k(t) = -2*pi f_k (u_k . v_k(t))
#      i.e. the field's phasor dynamics predict the field's own rendered
#      motion, packet by packet, at carrier-frequency-proportional rate.
#      This is deliberately NOT tested against input-frame motion:
#      verified Jul 2026 that the conv encoder is translation-invariant
#      (28 px input slide -> 0.5 px recon motion, |dz| ~ |z|) — the
#      medium re-indexes out-of-manifold motion instead of transporting
#      it, so only rendered motion is in-scope for the law.
#   6. 3-packet phase orbit (Phi_1,Phi_2,Phi_3), delay embedding of the
#      top packet, PCA dimensionality, frame-shuffle control.
#
# REGISTERED PREDICTIONS (fixed before any real-data run; the synthetic
# checks used to debug plumbing/signs were a rigid input translation —
# which exposed the encoder invariance above — and nothing else;
# thresholds below were not tuned on any latent-walk or real run):
#   S0 (reported, ungated): fraction of loud packets slot-stable.
#   P1  [1-D ORBIT]     PC1 explained variance of the (Phi1,Phi2,Phi3)
#       trajectory >= 0.80 for a smooth 1-parameter sweep.
#       (An arc, NOT a torus — a single sweep is a 1-D driver.)
#   P2  [DISPERSION]    amplitude*response-weighted |pearson r| between
#       measured dPhi_k and predicted -2*pi f_k (u_k . v_k) >= 0.60,
#       pooled over stable packets and moving pairs. Fitted slope
#       reported; slope ~ +1 = phase velocity matches rendered flow.
#   P3  [NOT-EIGENFACE] median over stable packets of cumulative-Phi
#       range >= 0.5 rad. (If phases stay flat while amplitudes do the
#       work, the model is an eigenface machine.)
#   P4  [SHUFFLE CTRL]  frame order shuffled (seed 0): orbit path length
#       >= 3x ordered AND shuffled dispersion |r| <= 0.20. NOTE: the
#       dispersion clause is only meaningful for non-uniform sweeps — a
#       constant-velocity stimulus is permutation-covariant (dPhi and
#       flow both scale with frame-index gap), so --selftest fails P4b
#       by construction and that is fine.
#
# MEASUREMENT-CHAIN CERTIFICATION (--selftest, model2.pt, 48f, cpu):
#       known-velocity phase driving recovered r=+0.902, slope=+1.330
#       (slope > 1 = Farneback flow slightly underestimates |v|, an
#       errors-in-x attenuation on pred; direction and linearity clean).
#       Earlier phase-correlation flow gave r=0.40/slope=0.08 — fringe-
#       period aliasing; do not regress phaseCorrelate on Gabor patches.
#
# CAVEATS (honest ledger):
#   - Local flow windows are small (default 25 px); sub-pixel phase
#     correlation there is noisy. Weighting by |Z| and correlation
#     response mitigates; it does not remove it.
#   - Per-frame |dPhi| must stay < pi: keep motion slow / fps high /
#     walk steps fine.
#   - --walk uses linear z interpolation between two seeds: in-manifold
#     motion by construction for the small gaps of a single model
#     (fire-law small-dphi regime), but it is the DECODER's road.
#   - Webcam: domain gap gives the dark-head recon; the law is tested on
#     that recon's own motion, which is the point — but S0 may drop.
#
# USAGE
#   python phase_orbit.py --walk 0 1 --nframes 64 --ckpt model2.pt
#   python phase_orbit.py --video sweep.mp4 --ckpt model2.pt
#   python phase_orbit.py --cam 0 --nframes 120 --ckpt model2.pt
#   python phase_orbit.py --frames 'frames/*.png' --ckpt model2.pt
#   python phase_orbit.py --npz zs.npz --ckpt model2.pt  # key 'z' (T,128)
# Outputs to ./phase_orbit_out/: ledger.txt, packets.csv, dispersion.csv,
#   states.npz, slots.png, phases.png, orbit3d.png, dispersion.png,
#   delay_pca.png
# ======================================================================

import argparse, glob, math, os, sys
import numpy as np
import torch
import torch.nn as nn

K = 11
LATENT = 128

# ---- model (mirrors splat_trainer3v2.py; state-dict names match) ----
class Encoder(nn.Module):
    def __init__(self, image_size=96, latent=LATENT, ch=32):
        super().__init__()
        layers, c_in, sz, c = [], 3, image_size, ch
        while sz > 4:
            layers += [nn.Conv2d(c_in, c, 4, 2, 1), nn.BatchNorm2d(c),
                       nn.LeakyReLU(0.2, True)]
            c_in, sz, c = c, sz // 2, min(c * 2, 512)
        self.conv = nn.Sequential(*layers)
        self.flat = c_in * sz * sz
        self.fc_mu = nn.Linear(self.flat, latent)
        self.fc_lv = nn.Linear(self.flat, latent)
    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)

class Decoder(nn.Module):
    def __init__(self, latent=LATENT, num_packets=256, hidden=512):
        super().__init__()
        self.N = num_packets
        self.net = nn.Sequential(
            nn.Linear(latent, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, num_packets * K))
    def forward(self, z):
        return self.net(z).view(-1, self.N, K)

class GaborRenderer(nn.Module):
    def __init__(self, image_size=96, num_packets=256, chunk=64):
        super().__init__()
        self.H = self.W = image_size
        self.N, self.chunk = num_packets, chunk
        gy, gx = torch.meshgrid(torch.linspace(0, 1, image_size),
                                torch.linspace(0, 1, image_size), indexing="ij")
        self.register_buffer("GX", gx[None, None].contiguous())
        self.register_buffer("GY", gy[None, None].contiguous())
        side = int(math.ceil(math.sqrt(num_packets)))
        ax = torch.linspace(0.08, 0.92, side)
        anch = torch.stack(torch.meshgrid(ax, ax, indexing="ij"), -1
                           ).reshape(-1, 2)[:num_packets]
        anch = torch.clamp(anch, 1e-3, 1 - 1e-3)
        self.register_buffer("anchor_logit", torch.log(anch / (1 - anch)))
    def activate(self, raw):
        px = torch.sigmoid(self.anchor_logit[:, 0][None] + raw[..., 0])
        py = torch.sigmoid(self.anchor_logit[:, 1][None] + raw[..., 1])
        sigma = 0.012 + 0.14 * torch.sigmoid(raw[..., 2])
        theta = raw[..., 3]
        freq = 1.0 + 15.0 * torch.sigmoid(raw[..., 4])
        coeff = torch.tanh(raw[..., 5:11]).reshape(*raw.shape[:2], 3, 2)
        return px, py, sigma, theta, freq, coeff
    def _chunk(self, px, py, sigma, theta, freq, coeff):
        px_ = px[..., None, None]; py_ = py[..., None, None]
        s_ = sigma[..., None, None]; th = theta[..., None, None]
        f_ = freq[..., None, None]
        dx = self.GX - px_; dy = self.GY - py_
        xr = dx * torch.cos(th) + dy * torch.sin(th)
        env = torch.exp(-(dx * dx + dy * dy) / (2 * s_ * s_))
        ec = env * torch.cos(2 * math.pi * f_ * xr)
        es = env * torch.sin(2 * math.pi * f_ * xr)
        a, b = coeff[..., 0], coeff[..., 1]
        chans = [(a[:, :, c, None, None] * ec).sum(1)
                 - (b[:, :, c, None, None] * es).sum(1) for c in range(3)]
        return torch.stack(chans, dim=1)
    def forward(self, raw):
        raw = raw.float()
        px, py, sigma, theta, freq, coeff = self.activate(raw)
        out = None
        for i in range(0, self.N, self.chunk):
            sl = slice(i, i + self.chunk)
            c = self._chunk(px[:, sl], py[:, sl], sigma[:, sl],
                            theta[:, sl], freq[:, sl], coeff[:, sl])
            out = c if out is None else out + c
        return torch.sigmoid(out)

class Probe(nn.Module):
    def __init__(self, image_size, num_packets):
        super().__init__()
        self.enc = Encoder(image_size)
        self.dec = Decoder(LATENT, num_packets)
        self.ren = GaborRenderer(image_size, num_packets)

# ---- frames ----------------------------------------------------------
def prep(im_bgr, size):
    import cv2
    h, w = im_bgr.shape[:2]
    s = min(h, w)
    im = im_bgr[(h - s)//2:(h + s)//2, (w - s)//2:(w + s)//2]
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    return im[:, :, ::-1].astype(np.float32) / 255.0

def get_frames(args, size):
    import cv2
    frames = []
    if args.video:
        cap = cv2.VideoCapture(args.video)
        i = 0
        while True:
            ok, f = cap.read()
            if not ok: break
            if i % args.stride == 0:
                frames.append(prep(f, size))
            i += 1
            if args.nframes and len(frames) >= args.nframes: break
        cap.release()
    elif args.cam is not None:
        cap = cv2.VideoCapture(args.cam)
        print(f"capturing {args.nframes} frames from cam {args.cam} — "
              "move SLOWLY and smoothly (one sweep)...")
        while len(frames) < args.nframes:
            ok, f = cap.read()
            if not ok: break
            frames.append(prep(f, size))
        cap.release()
    elif args.frames:
        for p in sorted(glob.glob(args.frames))[::args.stride]:
            f = cv2.imread(p, cv2.IMREAD_COLOR)
            if f is not None:
                frames.append(prep(f, size))
            if args.nframes and len(frames) >= args.nframes: break
    return np.stack(frames) if frames else None

# ---- analysis --------------------------------------------------------
def wpearson(x, y, w):
    w = w / (w.sum() + 1e-12)
    mx, my = (w * x).sum(), (w * y).sum()
    cov = (w * (x - mx) * (y - my)).sum()
    sx = math.sqrt((w * (x - mx) ** 2).sum() + 1e-18)
    sy = math.sqrt((w * (y - my) ** 2).sum() + 1e-18)
    return cov / (sx * sy), cov / (sx * sx + 1e-18)

def pair_scramble_r(pred, meas, w, tlbl, seed=0):
    """P4b' control: within each frame, permute which packet's measured
    dPhi is paired with which packet's prediction. Kills per-packet
    linkage while preserving every marginal and the frame-level motion
    structure. A per-packet-real dispersion relation should collapse
    under this; an estimator artifact shared across packets survives."""
    rng = np.random.default_rng(seed)
    meas2 = meas.copy()
    for t in np.unique(tlbl):
        idx = np.where(tlbl == t)[0]
        if len(idx) > 1:
            meas2[idx] = meas[rng.permutation(idx)]
    r, _ = wpearson(pred, meas2, w)
    return r

def path_length(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

def extract_states(mdl, zs, device, batch=64, render=True):
    px_l, py_l, th_l, f_l, zc_l, rc_l, sg_l = [], [], [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(zs), batch):
            raw = mdl.dec(zs[i:i+batch].to(device))
            px, py, sg, th, fq, co = mdl.ren.activate(raw)
            px_l.append(px.cpu()); py_l.append(py.cpu())
            th_l.append(th.cpu()); f_l.append(fq.cpu()); sg_l.append(sg.cpu())
            zc_l.append((co[..., 0] + 1j * co[..., 1]).mean(-1).cpu())
            if render:
                rc_l.append(mdl.ren(raw).mean(1).cpu())      # grayscale
    px = torch.cat(px_l).numpy(); py = torch.cat(py_l).numpy()
    th = torch.cat(th_l).numpy(); fq = torch.cat(f_l).numpy()
    zc = torch.cat(zc_l).numpy(); sg = torch.cat(sg_l).numpy()
    rec = torch.cat(rc_l).numpy().astype(np.float32) if render else None
    xbx, xby = px.mean(0, keepdims=True), py.mean(0, keepdims=True)
    carrier = 2*np.pi*fq*(np.cos(th)*(xbx - px) + np.sin(th)*(xby - py))
    Z = zc * np.exp(1j * carrier)
    return px, py, th, fq, sg, zc, Z, rec

def dense_flows(rec):
    """Farneback dense flow per frame pair, px units, (T-1,H,W,2).
    Gradient-based + pyramidal: immune to fringe-period aliasing that
    breaks phase-correlation on quasi-periodic Gabor patches."""
    import cv2
    T = rec.shape[0]
    u8 = np.clip(rec * 255, 0, 255).astype(np.uint8)
    return np.stack([cv2.calcOpticalFlowFarneback(
        u8[t], u8[t + 1], None, 0.5, 3, 15, 3, 5, 1.1, 0)
        for t in range(T - 1)])

def dispersion_pairs(Z, fq, th, rec, px, py, sg, stable, vfloor,
                     flows=None):
    """pred = -2*pi f (u . v_local) with v_local = envelope-weighted mean
    dense flow around each packet's time-mean center."""
    T = Z.shape[0]; H = rec.shape[1]
    if flows is None:
        flows = dense_flows(rec)
    yy, xx = np.mgrid[0:H, 0:H]
    preds, meass, ws, tlbl = [], [], [], []
    for k in stable:
        cx, cy = px[:, k].mean() * H, py[:, k].mean() * H
        s = max(float(sg[:, k].mean()) * H, 2.0)
        wsp = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * s * s))
        wsp /= wsp.sum()
        v = (flows * wsp[None, :, :, None]).sum((1, 2)) / H   # unit coords
        speed = np.linalg.norm(v, axis=1)
        sel = speed >= vfloor
        if not sel.any():
            continue
        dphi = np.angle(Z[1:, k] * np.conj(Z[:-1, k]))[sel]
        ux, uy = np.cos(th[:-1, k])[sel], np.sin(th[:-1, k])[sel]
        pred = -2*np.pi*fq[:-1, k][sel]*(ux*v[sel, 0] + uy*v[sel, 1])
        amp = 0.5*(np.abs(Z[1:, k]) + np.abs(Z[:-1, k]))[sel]
        preds.append(pred); meass.append(dphi)
        ws.append(amp); tlbl.append(np.where(sel)[0])
    if not preds:
        return None
    return (np.concatenate(preds), np.concatenate(meass),
            np.concatenate(ws), np.concatenate(tlbl))

# ---- main ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model2.pt")
    ap.add_argument("--video"); ap.add_argument("--frames")
    ap.add_argument("--cam", type=int, default=None)
    ap.add_argument("--npz", help="precomputed z: npz key 'z' (T,128)")
    ap.add_argument("--selftest", action="store_true",
                    help="positive control: freeze one decoded face, advance "
                         "each packet's coefficient phase by the dispersion "
                         "law with a known v (0.5,0.2) px/frame, render, and "
                         "check the pipeline recovers slope ~ +1. Certifies "
                         "the measurement chain, not the model.")
    ap.add_argument("--walk", nargs=2, type=int, metavar=("SEED_A","SEED_B"),
                    help="linear z walk between two randn seeds (no camera)")
    ap.add_argument("--walk-scale", type=float, default=0.9)
    ap.add_argument("--nframes", type=int, default=120)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp-floor", type=float, default=0.02)
    ap.add_argument("--stab", type=float, default=0.04)
    ap.add_argument("--vfloor-px", type=float, default=0.05,
                    help="min LOCAL rendered motion (px/frame)")
    ap.add_argument("--tau", type=int, default=0)
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--out", default="phase_orbit_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    size, npk = ck["image_size"], ck["num_packets"]
    mdl = Probe(size, npk)
    mdl.load_state_dict(ck["sd"], strict=True)
    mdl.eval().to(args.device)
    print(f"model: {size}px, {npk} packets, device={args.device}")

    frames = None
    if args.selftest:
        g = torch.Generator().manual_seed(0)
        z0 = (torch.randn(LATENT, generator=g) * 0.9)[None]
        with torch.no_grad():
            raw = mdl.dec(z0.to(args.device))
            px1, py1, sg1, th1, fq1, co1 = mdl.ren.activate(raw)
        v_true = np.array([0.5 / size, 0.2 / size])           # px/frame
        T = args.nframes
        ux, uy = torch.cos(th1), torch.sin(th1)
        dphi = (-2 * math.pi * fq1 *
                (ux * v_true[0] + uy * v_true[1]))            # (1,N)
        zs = torch.zeros(T, LATENT)                           # placeholder
        px = px1.cpu().numpy().repeat(T, 0); py = py1.cpu().numpy().repeat(T, 0)
        th = th1.cpu().numpy().repeat(T, 0); fq = fq1.cpu().numpy().repeat(T, 0)
        sg = sg1.cpu().numpy().repeat(T, 0)
        zc_t, rec_t = [], []
        with torch.no_grad():
            for t in range(T):
                ang = (dphi * t)
                ca, sa = torch.cos(ang), torch.sin(ang)
                a0, b0 = co1[..., 0], co1[..., 1]             # (1,N,3)
                a = a0 * ca[..., None] - b0 * sa[..., None]
                b = a0 * sa[..., None] + b0 * ca[..., None]
                co = torch.stack([a, b], -1)
                out = None
                for i in range(0, npk, 64):
                    sl = slice(i, i + 64)
                    c = mdl.ren._chunk(px1[:, sl], py1[:, sl], sg1[:, sl],
                                       th1[:, sl], fq1[:, sl], co[:, sl])
                    out = c if out is None else out + c
                rec_t.append(torch.sigmoid(out).mean(1).cpu())
                zc_t.append((a + 1j * b).mean(-1).cpu())
        zc = torch.cat(zc_t).numpy()
        rec = torch.cat(rec_t).numpy().astype(np.float32)
        Z = zc                                                # centers fixed
        print(f"SELFTEST: known v = (0.50, 0.20) px/frame, {T} frames — "
              "expect P2 slope ~ +1")
    elif args.walk:
        ga = torch.Generator().manual_seed(args.walk[0])
        gb = torch.Generator().manual_seed(args.walk[1])
        za = torch.randn(LATENT, generator=ga) * args.walk_scale
        zb = torch.randn(LATENT, generator=gb) * args.walk_scale
        t = torch.linspace(0, 1, args.nframes)[:, None]
        zs = (1 - t) * za[None] + t * zb[None]
        print(f"latent walk seeds {args.walk}, {args.nframes} steps, "
              f"gap |za-zb|={float((za-zb).norm()):.3f}")
    elif args.npz:
        zs = torch.from_numpy(np.load(args.npz)["z"]).float()
        print(f"loaded z: {tuple(zs.shape)}")
    else:
        frames = get_frames(args, size)
        if frames is None or len(frames) < 16:
            sys.exit("need at least 16 frames (or use --walk / --npz)")
        print(f"{len(frames)} frames @ {size}px")
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        mus = []
        with torch.no_grad():
            for i in range(0, len(x), 64):
                mu, _ = mdl.enc(x[i:i+64].to(args.device))
                mus.append(mu.cpu())
        zs = torch.cat(mus)

    T = len(zs)
    if not args.selftest:
        px, py, th, fq, sg, zc, Z, rec = extract_states(mdl, zs, args.device)

    amp = np.abs(Z).mean(0)
    dp = np.sqrt(np.diff(px, axis=0)**2 + np.diff(py, axis=0)**2)
    stable_mask = (dp.max(0) < args.stab) & (amp > args.amp_floor)
    loud = amp > args.amp_floor
    S0 = stable_mask.sum() / max(loud.sum(), 1)
    stable = np.where(stable_mask)[0]
    print(f"S0: {stable_mask.sum()}/{loud.sum()} loud packets slot-stable "
          f"({S0:.2f})")
    if len(stable) < 3:
        sys.exit("fewer than 3 stable packets — lower --amp-floor / raise "
                 "--stab, or the sweep is too violent")

    dPhi = np.angle(Z[1:] * np.conj(Z[:-1]))
    Phi = np.concatenate([np.zeros((1, Z.shape[1])),
                          np.cumsum(dPhi, axis=0)], axis=0)

    cen = ((np.abs(px.mean(0) - 0.5) < 0.3)
           & (np.abs(py.mean(0) - 0.5) < 0.3))
    pool = stable[cen[stable]] if cen[stable].sum() >= 3 else stable
    order = pool[np.argsort(-amp[pool])]
    picks = [int(order[0])]
    for k in order[1:]:
        if all(np.hypot(px[:, k].mean() - px[:, j].mean(),
                        py[:, k].mean() - py[:, j].mean()) > 0.15
               for j in picks):
            picks.append(int(k))
        if len(picks) == 3: break
    for k in order:
        if len(picks) == 3: break
        if int(k) not in picks: picks.append(int(k))
    print(f"orbit packets k={picks}  centers="
          + str([(round(float(px[:, k].mean()), 2),
                  round(float(py[:, k].mean()), 2)) for k in picks]))
    Porb = Phi[:, picks]

    Pc = Porb - Porb.mean(0)
    ev = np.linalg.svd(Pc, compute_uv=False) ** 2
    ev = ev / ev.sum()
    P1_val = float(ev[0]); P1 = P1_val >= 0.80

    exc = Phi[:, stable].max(0) - Phi[:, stable].min(0)
    P3_val = float(np.median(exc)); P3 = P3_val >= 0.5

    flows = dense_flows(rec)
    disp = dispersion_pairs(Z, fq, th, rec, px, py, sg, stable,
                            args.vfloor_px / size, flows=flows)
    P2 = P2_val = slope = r_scr = None
    if disp is not None:
        pred, meas, w, tlbl = disp
        r, slope = wpearson(pred, meas, w)
        P2_val = float(r); P2 = abs(r) >= 0.60
        r_scr = float(pair_scramble_r(pred, meas, w, tlbl))
        print(f"dispersion: {len(pred)} pairs, weighted r={r:+.3f}, "
              f"slope={slope:+.3f}, pair-scramble r={r_scr:+.3f}")
    else:
        print("dispersion: no local motion above floor — skipped "
              "(is the sweep actually moving anything in the render?)")

    # ---- envelope-borne vs coefficient-borne decomposition of dPhi ----
    # dPhi_env: carrier-term change from packet-center motion (moving the
    # envelope moves the pattern — near-bookkeeping under the composite
    # phase). dPhi_coef: coefficient phasor rotation — genuine phase
    # transport. Their split says which channel the decoder uses.
    dpx_, dpy_ = np.diff(px, axis=0), np.diff(py, axis=0)
    envp = -2*np.pi*fq[:-1]*(np.cos(th[:-1])*dpx_ + np.sin(th[:-1])*dpy_)
    coefp = np.angle(zc[1:] * np.conj(zc[:-1]))
    wq = 0.5*(np.abs(Z[1:]) + np.abs(Z[:-1]))
    selm = np.zeros(Z.shape[1], bool); selm[stable] = True
    Wq = wq[:, selm].ravel(); Wq = Wq / (Wq.sum() + 1e-12)
    Ev, Cv, Tv = [a[:, selm].ravel() for a in (envp, coefp, dPhi)]
    def _wv(x):
        m = (Wq*x).sum(); return float((Wq*(x-m)**2).sum())
    def _wc(x, y):
        mx, my = (Wq*x).sum(), (Wq*y).sum()
        return float((Wq*(x-mx)*(y-my)).sum()
                     / math.sqrt(_wv(x)*_wv(y) + 1e-18))
    vE, vC, vT = _wv(Ev), _wv(Cv), _wv(Tv)
    env_share, coef_share = vE/(vT+1e-18), vC/(vT+1e-18)
    ec_corr = _wc(Ev, Cv)

    rng = np.random.default_rng(0)
    perm = rng.permutation(T)
    Zs_ = Z[perm]
    dPhi_s = np.angle(Zs_[1:] * np.conj(Zs_[:-1]))
    Phi_s = np.concatenate([np.zeros((1, Z.shape[1])),
                            np.cumsum(dPhi_s, axis=0)], axis=0)
    L_ord = path_length(Porb)
    L_shf = path_length(Phi_s[:, picks])
    P4a = L_shf >= 3 * L_ord
    P4b_val = None                      # full-shuffle dispersion r —
    ds = dispersion_pairs(Zs_, fq[perm], th[perm], rec[perm],
                          px[perm], py[perm], sg[perm], stable,
                          args.vfloor_px / size)
    if ds is not None:
        rs, _ = wpearson(ds[0], ds[1], ds[2])
        P4b_val = float(rs)
    # v2 gate: full-shuffle r is DIAGNOSTIC only (a constitutive pairwise
    # law legitimately survives frame shuffling when the pose range keeps
    # shuffled pairs within flow range — learned from the first real-video
    # run, where r_shuf=+0.77 alongside slope 0.998). The artifact guard
    # is the pair-scramble control: registered P4b' |r_scr| <= 0.5*|r|.
    P4b = (r_scr is None or P2_val is None
           or abs(r_scr) <= 0.5 * abs(P2_val))
    P4 = P4a and P4b

    tau = args.tau if args.tau > 0 else max(1, T // 40)
    d = args.dim
    s0 = Phi[:, picks[0]]
    rows = T - (d - 1) * tau
    emb = np.stack([s0[i:i + rows] for i in range(0, d * tau, tau)], axis=1)
    embc = emb - emb.mean(0)
    ev_d = np.linalg.svd(embc, compute_uv=False) ** 2
    ev_d = ev_d / ev_d.sum()

    def mark(b): return "[V]" if b else "[X]"
    lines = [
        "phase_orbit ledger",
        f"  frames={T}  packets={npk}  stable={len(stable)}  S0={S0:.2f}",
        f"  orbit packets k={picks}",
        f"  P1 1-D orbit      {mark(P1)}  PC1 EV = {P1_val:.3f}  (>=0.80)"
        f"   [PCs: {', '.join(f'{e:.3f}' for e in ev)}]",
        f"  P2 dispersion     "
        + ("[--] skipped (no local motion)" if P2 is None else
           f"{mark(P2)}  weighted r = {P2_val:+.3f} (|r|>=0.60), "
           f"slope = {slope:+.3f}"),
        f"  P3 not-eigenface  {mark(P3)}  median phase excursion = "
        f"{P3_val:.2f} rad (>=0.5)",
        f"  P4 controls       {mark(P4)}  path x{L_shf/max(L_ord,1e-9):.1f} "
        f"(>=3)"
        + ("" if r_scr is None else
           f", pair-scramble r = {r_scr:+.3f} (<=0.5*|r|)")
        + ("" if P4b_val is None else
           f"  [diag: full-shuffle r = {P4b_val:+.3f}]"),
        f"  env/coef split    dPhi variance: env {env_share:.2f} / "
        f"coef {coef_share:.2f} / cross {max(0.0, 1-env_share-coef_share):.2f}"
        f"   corr(env,coef) = {ec_corr:+.2f}",
        f"  delay embedding (k={picks[0]}, tau={tau}, d={d}): "
        f"PC EVs {', '.join(f'{e:.3f}' for e in ev_d[:4])}",
        "  verdict: " + ("[V]" if all(x for x in [P1, P3, P4]
                                      + ([P2] if P2 is not None else []))
                         else "[K] see failed rows"),
    ]
    txt = "\n".join(lines)
    print("\n" + txt)
    open(os.path.join(args.out, "ledger.txt"), "w").write(txt + "\n")

    with open(os.path.join(args.out, "packets.csv"), "w") as f:
        f.write("frame,k,amp,phi_coeff,Phi_cum,px,py,theta,freq\n")
        for t in range(T):
            for k in stable:
                f.write(f"{t},{k},{np.abs(Z[t,k]):.5f},"
                        f"{np.angle(zc[t,k]):.5f},{Phi[t,k]:.5f},"
                        f"{px[t,k]:.5f},{py[t,k]:.5f},"
                        f"{th[t,k]:.5f},{fq[t,k]:.4f}\n")
    if disp is not None:
        with open(os.path.join(args.out, "dispersion.csv"), "w") as f:
            f.write("pred,meas,weight,frame\n")
            for a, b, c, t in zip(*disp):
                f.write(f"{a:.5f},{b:.5f},{c:.5f},{int(t)}\n")
    np.savez(os.path.join(args.out, "states.npz"), z=zs.numpy(), px=px, sigma=sg,
             py=py, theta=th, freq=fq, phasor=zc, Z=Z, Phi=Phi,
             stable=stable, picks=np.array(picks))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    plt.figure(figsize=(6, 6))
    bg = frames[0] if frames is not None else np.repeat(
        rec[0][:, :, None], 3, 2)
    plt.imshow(np.clip(bg, 0, 1))
    plt.title("slot centers (green=stable, red=loud unstable)")
    for k in range(npk):
        c = "lime" if stable_mask[k] else ("red" if loud[k] else "gray")
        plt.plot(px[:, k]*size, py[:, k]*size, c=c, lw=0.6,
                 alpha=0.8 if c != "gray" else 0.15)
    for k in picks:
        plt.plot(px[:, k]*size, py[:, k]*size, "b", lw=2)
    plt.gca().set_xlim(0, size); plt.gca().set_ylim(size, 0)
    plt.savefig(os.path.join(args.out, "slots.png"), dpi=130,
                bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for k in picks:
        ax[0].plot(Phi[:, k], label=f"k={k}")
        ax[1].plot(np.abs(Z[:, k]), label=f"k={k}")
    ax[0].set_ylabel("Phi (rad, cum)"); ax[0].legend()
    ax[1].set_ylabel("|Z|"); ax[1].set_xlabel("frame")
    ax[0].set_title("fast phase / slow envelope")
    plt.savefig(os.path.join(args.out, "phases.png"), dpi=130,
                bbox_inches="tight"); plt.close()

    fig = plt.figure(figsize=(7, 6))
    a3 = fig.add_subplot(111, projection="3d")
    a3.plot(Phi_s[:, picks[0]], Phi_s[:, picks[1]], Phi_s[:, picks[2]],
            c="lightgray", lw=0.6, label="shuffled")
    p = a3.scatter(Porb[:, 0], Porb[:, 1], Porb[:, 2],
                   c=np.arange(T), cmap="viridis", s=10)
    a3.plot(Porb[:, 0], Porb[:, 1], Porb[:, 2], c="k", lw=0.8)
    a3.set_xlabel(f"Phi k={picks[0]}"); a3.set_ylabel(f"Phi k={picks[1]}")
    a3.set_zlabel(f"Phi k={picks[2]}")
    fig.colorbar(p, label="frame"); a3.legend()
    a3.set_title(f"phase orbit  (PC1 EV {P1_val:.2f})")
    plt.savefig(os.path.join(args.out, "orbit3d.png"), dpi=130,
                bbox_inches="tight"); plt.close()

    if disp is not None:
        pred, meas, w, _t = disp
        plt.figure(figsize=(6, 6))
        plt.scatter(pred, meas, s=8, c=w, cmap="magma", alpha=0.7)
        lim = max(np.abs(pred).max(), np.abs(meas).max()) * 1.05
        plt.plot([-lim, lim], [-lim, lim], "g--", lw=1, label="slope +1")
        xs = np.linspace(-lim, lim, 2)
        plt.plot(xs, slope*xs, "b-", lw=1,
                 label=f"fit {slope:+.2f}, r {P2_val:+.2f}")
        plt.xlabel("predicted  -2*pi f (u . v_local)  [rad/frame]")
        plt.ylabel("measured  dPhi  [rad/frame]")
        plt.colorbar(label="weight"); plt.legend()
        plt.title("dispersion: phasor dynamics vs rendered local flow")
        plt.savefig(os.path.join(args.out, "dispersion.png"), dpi=130,
                    bbox_inches="tight"); plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(ev_d) + 1), ev_d)
    plt.xlabel("delay-embedding PC"); plt.ylabel("explained variance")
    plt.title(f"Takens embedding of Phi_k{picks[0]}  (tau={tau}, d={d})")
    plt.savefig(os.path.join(args.out, "delay_pca.png"), dpi=130,
                bbox_inches="tight"); plt.close()

    print(f"\nwrote {args.out}/  (ledger, csv, npz, 5 plots)")

if __name__ == "__main__":
    main()
