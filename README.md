# Tiny Avatar

A ~22 MB face model you can train on yourself and drive live from a webcam —
rendered by nothing but wave interference.

A small VAE maps a 128-dimensional latent to a few hundred **Gabor wave
packets** (position, scale, orientation, frequency, and a complex amplitude
per color channel). The image is the sum of those packets' ripples pushed
through a sigmoid. No pixel buffers, no convolutions in the decoder — the
face *is* the interference pattern.

The avatar runs on **phase-transport pursuit**: between encoder keyframes,
each packet's complex amplitude is rotated along the shortest arc instead of
crossfaded. Crossfading complex amplitudes cancels the wave mid-path (the
image dissolves into "fire"); rotating them keeps amplitude constant while
the ripple physically glides. That is the one idea this repo exists to
demonstrate, and it was tested before it was trusted (see **Ledger** below).

![Training Studio](avatar_studio.png)

## What's in the repo

| file | what it is |
|---|---|
| `tiny_avatar.py` | the studio app — dataset prep, training, avatar driving, one window |
| `splat_trainer3v2.py` | the trainer (standalone CLI; the app wraps it as a subprocess) |
| `model2.pt` | a CelebA-trained checkpoint (30k epochs)(96 px / 256 packets) so you can try the avatar without training anything |

## Install

```
pip install torch PyQt6 opencv-python numpy psutil
pip install pynvml        # optional: GPU % readout in the training tab
python tiny_avatar.py
```

Python 3.10+. CUDA strongly recommended for training; the avatar itself
runs even on CPU (the model is tiny).

## Try it in one minute (no training)

1. Start the app, go to **Avatar Driver**.
2. The bundled `model2.pt` is auto-detected (any `.pt` next to the app or
   under `runs/` shows up in the dropdown).
3. Click **Latent walk** — the model surfs its own face manifold using
   phase-transport pursuit. This is the "surf that never melts" demo.
4. Or click **Start webcam**. Expect this, honestly: with the CelebA model
   your reconstruction will be a **blurry dark head that tracks your pose**.
   That's a domain gap, not a bug — the model was trained on aligned,
   evenly-lit celebrity crops, and your webcam is neither. Pose survives
   the gap; identity doesn't. To get an avatar that actually looks like
   you, train on yourself:

## Train on yourself

**1 — Record.** 1–2 minutes of yourself talking, turning your head through
angles, changing expression, with a little lighting variation. One person
only. Coverage of pose + expression is what lets the model lock and track.
Landscape orientation is safest (portrait phone clips carry a rotation tag;
the app reads it and un-rotates, but if the preview still shows you
sideways, re-shoot in landscape).

**2 — Dataset Prep tab.** Point it at the video. Face-detect crop is on by
default and recommended — without it your face is a small blob in a wide
frame. Watch the preview: if the crop locks onto the wrong thing (a shadow,
a poster), stop, uncheck face-detect, and re-run with the center crop.
You want at least a few hundred frames; ~2000 is comfortable.

**3 — Training Studio tab.** Set `data_dir` to your frames folder, pick an
out dir, hit **Take the pulse** — it checks your VRAM against the dataset
and renderer and engages the `--disk` memmap fallback automatically if the
cache won't fit (that's how 128 px / 512 packets trains on a 12 GB card).
Then **Start training**.

- Training takes **hours, not minutes**. The recon/sample previews update
  every log step so you can watch it learn.
- Left preview: real frames (top rows) vs reconstructions. Right: pure
  samples from the prior — these look like woodgrain static early on and
  slowly become faces. That order is normal: the model commits envelopes
  first, carrier phase snaps in later.
- Low `beta` (default 0.001) keeps a single identity sharp. High beta on a
  small single-person dataset will posterior-collapse to gray.
- You can **Stop** any time — a checkpoint is saved every log step, and
  the app detects it and offers **Resume** (the preprocessing cache is
  also reused, so restarts are instant).

**4 — Avatar Driver tab.** Select your new checkpoint, **Start webcam**.
Controls:

- **mode** — `phase pursuit` (the transport mechanism), `lerp pursuit`
  (baseline, for comparison), `direct` (re-encode every frame; jittery,
  full cost). Flip between them live and watch what fast head motion does.
- **keyframe every N frames** — how often the encoder runs. Everything in
  between is pure transport.
- **pursuit alpha** — how big a fractional step toward the latest keyframe
  each display frame takes. Higher = snappier, lower = smoother.
- **normalize input** — pushes webcam brightness/contrast toward
  face-dataset statistics; helps the domain gap.

## Using the trainer without the GUI

The app is only a cockpit; the trainer is a normal CLI and stays
authoritative:

```
python splat_trainer3v2.py --data_dir faces1 --out runs/me \
       --image_size 128 --num_packets 512 --beta 0.001 --disk
```

First run builds a one-time uint8 `.npy` cache of the dataset. `--disk`
keeps that cache as a disk memmap with batch-only reads instead of loading
it resident — VRAM → RAM → disk, in order of what fits.

## Ledger — what is measured vs. what is a demo

This project's rule is: do not hype, do not lie, just show. Registered
thresholds, then runs, then verdicts.

**Certified** (8 random face pairs, thresholds written before running,
`splat_avatar_gate.py`):

- Transport mid-path frames are **sharper than the decoder's own latent
  road** at every t — 8/8.
- Transport stays within 0.35× of the decoder road even across **full
  identity swaps** (typical deviation ratio 0.02–0.06) — 8/8. A single
  transport segment covers far more distance than any sane keyframe step,
  so keyframe rate is set by responsiveness, not by transport breaking.
- Scrambling target phases breaks road agreement ~10× — 8/8. Coherent
  phase is the mechanism, not a coincidence.
- **Honest miss:** the predicted lerp "fire" between same-model faces was
  confirmed on only 4/8 pairs — between nearby faces the phases correlate
  and plain crossfade is milder than the theory expected. Transport's
  in-manifold value is amplitude discipline plus the sharpness margin; its
  out-of-manifold value is not dying. The catastrophic fire lives at large
  phase differences and off-manifold moves.

**Demo, not certificate:**

- The **pursuit scheme** itself (fractional transport steps toward a moving
  keyframe) rides strictly inside the certified geodesics but has no gate
  of its own yet.
- The pulse check's renderer estimate is a labeled heuristic.

**Known limitations:**

- The webcam encoder has a real domain gap on out-of-distribution input
  (lighting, framing) — single-identity training is the fix, input
  normalization is the band-aid.
- Face detection is Haar cascade: fast, dumb, upright-only. It can latch
  onto face-like non-faces; the extraction preview exists so you catch
  that before training does.
- This will not out-render modern talking-head or diffusion avatars, and
  isn't trying to. The trade it makes: a model measured in single-digit
  megabytes, frames between keyframes produced by one rotation matrix,
  and every claim above backed by a CSV.

## License / provenance

Trainer, renderer math, transport tests, and app developed in an ongoing
human + AI collaboration (Anthropic's Claude models, Google Gemini, others)
at PerceptionLab. The bundled checkpoint was trained on CelebA — respect
the CelebA terms (non-commercial research) for that file; your own-face
models are yours.
