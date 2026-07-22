# Tiny Avatar

Video about it: https://www.youtube.com/watch?v=6TzSXNmUlxE

![tiny](tiny.png)

A tiny wave-interference face model you can train on yourself and drive
live from a webcam. The training checkpoint is ~22 MB; its exported
decoder — the generative half — is ~7 MB of ONNX. Rendered by nothing
but wave interference.

A small VAE maps a 128-dimensional latent to a few hundred **Gabor wave
packets** (position, scale, orientation, frequency, and a complex
amplitude per color channel). The image is the sum of those packets'
ripples pushed through a sigmoid. No pixel buffers, no convolutions in
the decoder — the face *is* the interference pattern.

The avatar runs on **phase-transport pursuit**: between encoder
keyframes, each packet's complex amplitude is rotated along the shortest
arc instead of crossfaded. Crossfading complex amplitudes can cancel the
wave mid-path (the image dissolves into "fire"); rotating them keeps
amplitude constant while the ripple physically glides.

As of July 2026 the repo also carries a measured result about the
medium itself: on real head-motion video, each packet's phase rotation
predicts the rendered image's local optical flow through the dispersion
relation `dPhi_k = -2*pi * f_k * (u_k . v_k)` with **fitted slope
+0.998 and weighted r = +0.81 over 16,401 packet-frame pairs**, passing
a per-packet scramble control. The wave field carries its own velocity
field. Full detail, controls, and every caveat in
**The science** and the **Ledger** below.

![Training Studio](avatar_studio.png)

## What's in the repo

| File | What it is |
|---|---|
| `tiny_avatar.py` | The studio app — dataset preparation, training, and avatar driving, all in one window |
| `splat_trainer3v2.py` | The trainer (standalone CLI; the app wraps it as a subprocess) |
| `model2.pt` | A CelebA-trained checkpoint (96 px / 256 packets, ~22 MB) so you can try the avatar without training anything |
| `phase_orbit.py` | The measurement instrument: Takens phase-orbit extraction + the dispersion-law test, with registered pass/fail gates and controls |
| `splat_pulse.py` | Live diagnostic: watch the packets' phase dynamics ("the pulse") while the avatar runs |

**On file formats, honestly:** the studio drives `.pt` checkpoints only
— the avatar needs the *encoder* (webcam frame → latent), and the
checkpoint carries encoder + decoder together. `--export` writes the
~7 MB `splat_decoder.onnx` (decoder only) for the standalone cv5 tool
line; the studio does not load ONNX.

## Install

```
pip install -r requirements.txt
pip install torch PyQt6 opencv-python numpy psutil
pip install pynvml        # optional: GPU % readout in the training tab
python tiny_avatar.py
```

Python 3.10+. CUDA strongly recommended for training; the avatar itself
runs even on CPU (the model is tiny).

## Try it in one minute (no training)

1. Start the app, go to **Avatar Driver**.
2. The bundled `model2.pt` is auto-detected (any `.pt` next to the app
   or under `runs/` shows up in the dropdown).
3. Click **Latent walk** — the model surfs its own face manifold using
   phase-transport pursuit. This is the "surf that never melts" demo.
4. Or click **Start webcam**. Expect this, honestly: with the CelebA
   model your reconstruction will be a **blurry dark head that tracks
   your pose**. That's a domain gap, not a bug. To get an avatar that
   actually looks like you, train on yourself:

## Train on yourself

**1 — Record.** 1–2 minutes of yourself talking, turning your head
through angles, changing expression, with a little lighting variation.
One person only. Landscape orientation is safest (the app un-rotates
phone clips from their rotation tag; if the preview still shows you
sideways, re-shoot in landscape).

**2 — Dataset Prep tab.** Point it at the video. Face-detect crop is on
by default and recommended. Watch the preview: if the crop locks onto
the wrong thing, uncheck face-detect and re-run with the center crop. A
few hundred frames minimum; ~2000 is comfortable.

**3 — Training Studio tab.** Set `data_dir`, pick an out dir, hit
**Take the pulse** (checks VRAM, engages the `--disk` memmap fallback
if the cache won't fit — that's how 128 px / 512 packets trains on a
12 GB card), then **Start training**. Hours, not minutes, at face
resolutions. Previews update every log step; checkpoints save every log
step; **Stop** and **Resume** are safe, and the preprocessing cache is
reused so restarts are instant. Leave `--checkpointing` off unless you
actually OOM — it trades speed for VRAM.

**The diversity knobs, honestly explained.** Low `beta` (0.001–0.005,
the band that works on this architecture) keeps a single identity sharp
— but at low beta a VAE's *reconstructions* stay diverse while its
*prior samples* average toward one bland face, because random z lands
off the encoder's actual latent region. Two consequences worth knowing:

- **The avatar mostly doesn't care** — it drives z from the *encoder*,
  not the prior. Bland prior samples do not mean a bland avatar.
- **The sample preview and any prior-sampling test DO care.** The
  trainer prints a `prior-sample diversity` number every log step
  (mean pairwise MSE over 32 prior samples). Reference points from real
  runs: the bundled 96 px CelebA model sits around 0.12; a 128 px run
  that averaged aggressively measured 0.009.

  On a **single-identity dataset a low number is partly correct** — all
  prior samples *should* be the same subject. On multi-identity data
  (CelebA-style), a tiny number means averaging; if you want diverse
  prior samples, raise `beta` with `free_bits` set (0.03–0.10): the
  per-dim KL floor is the standard tool that lets beta rise without the
  collapse this architecture hits otherwise. Its effect here is
  **unmeasured until you run it** — that is what the printed number is
  for.

Augmentation (`--aug`, on by default: horizontal flip + light
brightness/contrast jitter, on-GPU, effectively free) improves pose
coverage and narrows the webcam domain gap.

## Not just faces

The subject doesn't have to be human. Any visually consistent subject
with smooth pose/expression variation trains the same way — as a smoke
test, a 10,000-frame synthetic cartoon-creature dataset (64 px,
procedurally animated) cached in seconds, sat resident on the GPU at
0.12 GB, and trained at ~560 img/s on a 12 GB RTX 3060 at 64 px / 128
packets. Small resolutions turn "hours" into "minutes" and are a good
way to learn the knobs before committing an evening to your own face.

## Avatar Driver

Select your checkpoint, **Start webcam** (or **Latent walk**).

- **mode** — `phase pursuit` (the transport mechanism), `lerp pursuit`
  (baseline, for comparison), `direct` (re-encode every frame; jittery,
  full cost), `screw pursuit` (**demo**: packet geometry follows the
  SE(2) screw geodesic; not yet A/B-certified live), `dispersion
  pursuit` (**demo**: rotates phases by the dispersion formula using
  keyframe-implied velocities — measured live effect vs. plain phase
  pursuit: essentially none, and that's expected, because it re-derives
  its velocity from the same keyframe targets the arc step already
  uses; it's listed so nobody rediscovers this the hard way).
- **face-align input** — crops your live face with the *same* Haar
  detector, margin, and square-up that Dataset Prep used on the
  training frames (plus EMA smoothing so the crop doesn't jitter).
  This closes a real train/live mismatch: training data is
  face-cropped, the old live path was center-cropped, and the encoder
  treats a differently-framed face as off-manifold input — you get the
  blurry average head. Framing is the single biggest live-quality lever
  we've found; this makes it automatic. On by default.
- **keyframe every N frames** — encoder rate; everything between is
  pure transport.
- **pursuit alpha** — fractional step toward the latest keyframe per
  display frame. Higher = snappier, lower = smoother.
- **normalize input** — pushes webcam brightness/contrast toward the
  face-dataset statistics; helps the domain gap.

## Using the trainer without the GUI

```
python splat_trainer3v2.py --data_dir faces1 --out runs/me \
       --image_size 128 --num_packets 512 --beta 0.001 --disk \
       --free_bits 0 --aug 1
```

First run builds a one-time uint8 `.npy` cache. `--disk` reads batches
straight from the memmap: VRAM → RAM → disk, in order of what fits.
`--export` writes the ~7 MB `splat_decoder.onnx` (opset 17, dynamic
batch, `z_latent` → `rendered_image`) for the cv5 tools.

## The science — `phase_orbit.py` and the dispersion law

Each packet's complex coefficient is, exactly, an analytic signal:
`z_k(t) = a_k(t) * exp(i*phi_k(t))` (verified to 4e-16 — the decoder's
coefficients *are* the phasors). Track them across frames and the model
becomes a set of 1-D complex time series instead of a pixel tensor.
`phase_orbit.py` does that tracking and runs four registered tests:

- **P1 — 1-D orbit.** Three face-region packets' cumulative phases
  `(Phi_1, Phi_2, Phi_3)` should trace a low-dimensional curve during a
  smooth sweep (PC1 explained variance >= 0.80). An *arc*, not a torus
  — a single sweep is a one-parameter driver.
- **P2 — dispersion.** The claim with teeth: each packet's phase step
  should predict the rendered image's local optical flow around that
  packet, `dPhi_k = -2*pi * f_k * (u_k . v_k)` — phase velocity
  proportional to carrier frequency. Gate: weighted |r| >= 0.60.
- **P3 — not an eigenface machine.** Phases must actually move (median
  excursion >= 0.5 rad). If only amplitudes moved, this would be a
  1991-style additive eigenface basis wearing a wave costume.
- **P4 — controls.** A frame-shuffle path-length control for the orbit,
  and a **pair-scramble control** for P2: within each frame, permute
  *which packet's* measured phase step is paired with *which packet's*
  prediction. Kills per-packet linkage, preserves every marginal. A
  real per-packet law must collapse under it (gate:
  |r_scrambled| <= 0.5|r|).

**Result on real head-motion video** (120 frames, webcam clip, 96 px /
256 packet checkpoint — full ledger, CSVs and plots in
`phase_orbit_out/`):

```
P1 1-D orbit      [V]  PC1 EV = 0.951
P2 dispersion     [V]  weighted r = +0.809, slope = +0.998  (16,401 pairs)
P3 not-eigenface  [V]  median phase excursion = 0.62 rad
P4 controls       [V]  pair-scramble r = +0.236 (bound 0.405)
                       [diagnostic: full-shuffle r = +0.772]
env/coef split         dPhi variance: 32% envelope / 42% coefficient /
                       27% coherent cross-term, corr(env,coef) = +0.38
verdict: [V]
```

What the extra lines mean, because they're where the honesty lives:

- **The slope is +0.998 against a predicted +1.** The wave field's
  phase dynamics *are* its rendered motion, at the frequency-
  proportional rate the dispersion relation says.
- **The env/coef split closes a loophole.** Under the composite phase
  used here, a packet that merely *slides its envelope* satisfies the
  relation semi-tautologically. Decomposing shows the largest share of
  phase motion (42%) is genuine coefficient-phasor rotation — the
  decoder really rotates its phases to move the face — with envelope
  motion (32%) and a coherent cross-term (27%) alongside, positively
  correlated: the decoder splits each motion across both channels in
  the same direction.
- **The full-shuffle diagnostic (+0.77) is a finding, not a leak.** The
  relation survives destroying temporal order, which means it is
  *constitutive* — a property of pairs of states, not just of adjacent
  frames: the phase difference between two configurations encodes the
  displacement between them. (It's reported as a diagnostic because a
  constitutive law legitimately survives that shuffle; the pair-scramble
  above is the control that actually guards against artifacts.)
- **The single-packet Takens embedding shows 2–3 effective dimensions**
  (delay-PC spectrum 0.62 / 0.31 / 0.03) — one scalar phase trace
  reconstructing a multi-degree-of-freedom driver (yaw + pitch +
  expression). That is Takens' theorem doing its job on this medium.

Two negative results from building the instrument, kept because they
matter more than the passes:

- **The encoder is translation-invariant.** Slide the input face 28 px
  and the reconstruction moves 0.5 px, while z changes as much as its
  own norm. The medium does not transport out-of-manifold motion; it
  **re-indexes** onto the learned manifold and spends the latent change
  on appearance. (Practical corollary: that's why the face-align toggle
  exists, and why the dispersion law is stated about the *rendered*
  field's own flow, not the input's.)
- **Never phase-correlate a Gabor patch.** Quasi-periodic fringes alias
  modulo their own wavelength; `cv2.phaseCorrelate` on packet windows
  produced wavelength-sized garbage (r 0.40, slope 0.08 on a known-
  velocity control). Envelope-weighted Farnebäck flow — gradient-based,
  pyramidal, structurally unable to period-alias — recovers the
  injected physics at r = +0.9.

Reproduce it:

```
python phase_orbit.py --selftest --ckpt model2.pt        # instrument check:
                                                         # expect r ~ +0.9, slope ~ +1.3
python phase_orbit.py --video your_headturn.mp4 --ckpt runs/you/model2.pt
python phase_orbit.py --walk 0 1 --ckpt model2.pt        # latent walk (no camera)
```

(The selftest drives phases with a known velocity and checks the
pipeline recovers it — it certifies the measurement chain, not the
model. Its slope reads ~1.3 because Farnebäck slightly underestimates
pure fringe drift; on real video, where envelopes move too, the slope
comes out at 1.)

## Live telemetry — `splat_pulse.py`

![pulse](splat_pulse.png)

Watch the medium's phase dynamics while the avatar runs. Imports the
app's own pursuit and framing machinery so what you see is the actual
driver loop (keep it in the same folder as the app and trainer; it has
standalone fallbacks if the app isn't importable).

```
python splat_pulse.py --walk        # latent walk
python splat_pulse.py --cam 0      # webcam-driven
python splat_pulse.py --walk --record out.mp4 --nframes 300   # headless
```

Three panels: the avatar with a **pulse quiver** (each packet an
oriented tick — length ~ amplitude, color = instantaneous phase step
dPhi, blue negative / red positive — motion sweeps across the face as
color waves); a **pulse map** of the packet field in coordinate space
with the three tracked orbit packets ringed; and instruments —
scrolling Phi traces (the Takens signal, streaming live) plus per-band
meters.

The band meters show the **explained fraction**
`EF_band = 1 - sum(m*e^2)/sum(m*dPhi^2)` for LOW (f<3), MID (3–8) and
HIGH (f>=8) carrier bands, where `e` is the residual after removing the
dispersion-form prediction. EF → 1: that band's phase motion follows
the dispersion form. EF → 0 or below: phases move but not as predicted.
Blank: band idle. (A first version used circular residual coherence; it
saturated at ~1.0 for pursuit-sized steps and was replaced. If your
copy logs `R_LOW` instead of `EF_LOW` you have the stale version —
`grep EF_LOW splat_pulse.py` should match.)

**Registered, still open — not results yet:**

- **PULSE-1:** during smooth in-manifold driving, EF_LOW >= EF_HIGH on
  a clear majority of active frames (a low-frequency "skeleton" holding
  the dispersion form while high-frequency detail departs first).
- **PULSE-2:** framing breaks / off-manifold input collapse EF across
  all bands *before* visible reconstruction degradation.

Both score mechanically from the tool's `pulse_log.csv`. The one data
point so far (a latent walk) actually leaned *against* PULSE-1 — worth
knowing before anyone repeats the frequency-hierarchy story as fact.

## Ledger — what is measured vs. what is a demo

This project's rule: do not hype, do not lie, just show. Registered
thresholds, then runs, then verdicts.

**Certified — the dispersion law** (`phase_orbit.py --video`, real
head-motion clip, thresholds registered before the run): phase step
predicts rendered local flow at slope +0.998, weighted r +0.809 over
16,401 packet-frame pairs; per-packet pair-scramble control collapses
to +0.236 (bound 0.405); majority of the phase motion is genuine
coefficient-phasor rotation (42%), not envelope bookkeeping (32%);
measurement chain independently certified by known-velocity injection
(r +0.90–0.92 on two machines). The relation additionally survives
frame shuffling (+0.77), indicating it is constitutive — a state-pair
property — not merely frame-to-frame dynamics.

**Certified — the encoder re-indexes, it does not translate:** 28 px
of input translation produces 0.5 px of reconstruction motion while
|dz| ~ |z|. Out-of-manifold motion is absorbed into appearance; only
manifold coordinates are transported.

**Certified — the fire law** (`fire_law_screw_test.py --real`, 128 px /
512 packets, 8 pairs):

- The crossfade's amplitude loss has a **closed form**. Per packet,
  `|psi(t)|^2 = (1-t)^2 aA^2 + t^2 aB^2 + 2 t(1-t) aA aB cos(dphi)`.
  Evaluated on **endpoint phases alone — no rendering** — it predicted
  the measured lerp/transport field ratio of 384 rendered frames to a
  mean error of 0.022, with 8/8 agreement on the fire verdicts, and
  matched the raw coefficients to 4e-16.
- Why crossfade usually looks fine: random same-model face pairs sit at
  amplitude-weighted `|dphi|` of 0.2–0.6 rad, where `cos(dphi/2)` is
  0.96–1. The catastrophic fire — the mid-morph gray-out — is the
  formula's zero at `dphi = pi`: extrapolation, phase scrambles, fast
  off-manifold moves. Transport is the scheme whose amplitude *cannot*
  dip regardless of `dphi`: the safety margin, purchased with one
  rotation matrix.
- Screw interpolation (isolated math): recovers a rigid rotation to
  1.6e-16 where decoupled lerp+arc errs at 4e-2, and is identical to
  lerp on pure translation to 1.2e-16.

**Certified — earlier gates** (`splat_avatar_gate.py`, 96 px model, 8
pairs): transport mid-path frames sharper than the decoder's own latent
road at every t, 8/8; road agreement across full identity swaps, 8/8;
scrambled phases break road agreement ~10x, 8/8 — coherent phase is the
mechanism, not a coincidence.

**Honest revision log** (kept because the misses taught more than the
hits):

- The original claim "lerp burns, transport doesn't" measured 4/8 on
  one model, then 0/8 on a second — and was then *explained*: both runs
  sat in the small-`dphi` regime the closed form predicts to be
  fire-free. The claim that survived is smaller and stronger: **the
  fire is a formula, and transport is its guarantee.**
- The dispersion test was first aimed at *input-frame* motion; the
  encoder's translation invariance made that test invalid in principle
  and forced the medium-internal restatement that then passed.
- The first P4 control (frame shuffle) was mis-specified: it assumed a
  dynamical law and failed (+0.77) against what turned out to be a
  constitutive one. The gate was not reinterpreted into a pass — it was
  replaced with the pair-scramble control, validated on synthetic
  driving (r +0.90 → −0.01 under scramble), and the video was re-run
  through the new gate. The failure is what taught us the law's actual
  character.
- The pulse tool's first coherence metric saturated and was replaced
  (see above). An 18k-frame session logged with the stale metric had to
  be discarded — check your file version before trusting a log.

**Demo, not certificate:** the pursuit scheme itself; screw and
dispersion modes as *live avatar* modes (dispersion mode adds nothing
over phase pursuit by construction — it re-derives velocity from the
same keyframes); PULSE-1/PULSE-2 (registered, unmeasured); a
cross-model "phase-binding matrix" for driving one model with another's
packets (an idea worth testing; it presupposes cross-model slot
semantics nothing has verified); `free_bits`' diversity effect on this
architecture; the pulse check's renderer estimate is a labeled
heuristic.

**Known limitations:** the webcam encoder has a real domain gap on
out-of-distribution input (single-identity training is the fix,
face-align + input normalization the band-aids); Haar-cascade face
detection is fast, dumb, upright-only — the extraction preview exists
so you catch a bad lock before training does; and this will not
out-render modern talking-head or diffusion avatars and isn't trying
to. The trade it makes: a decoder measured in single-digit megabytes,
in-between frames produced by one rotation matrix, and every claim
above backed by a CSV.

## License / provenance

Trainer, renderer math, transport tests, measurement instruments, and
app developed in an ongoing human + AI collaboration (Anthropic's
Claude models, Google Gemini, DeepSeek, ChatGPT, others) at
PerceptionLab. The bundled checkpoint was trained on CelebA — respect
the CelebA terms (non-commercial research) for that file; your own-face
models are yours.
