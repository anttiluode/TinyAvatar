#!/usr/bin/env python3
# =============================================================================
# tiny_avatar.py — TINY AVATAR studio
#
# One PyQt6 app wrapping the whole splat pipeline:
#   Tab 1  Home            what this is
#   Tab 2  Dataset Prep    video -> face-cropped frames, or drop an image folder
#   Tab 3  Training Studio runs your trainer as a SUBPROCESS (OOM can never
#                          take the GUI down), parses its log lines, shows
#                          live recon/sample previews + GPU/RAM pulse,
#                          detects resumable runs, pulse-checks VRAM before
#                          launch and auto-engages --disk when supported
#   Tab 4  Avatar Driver   the certified transport pursuit (from
#                          avatar_driver.py) driving webcam or latent walk,
#                          rendered into the app itself
#
# TRAINER ADAPTER (the only coupling point):
#   the app looks for, in order:  splat_trainer3v2.py, splat_trainer3.py,
#   Splat_trainer2.py  in its own directory. It reads the chosen file's source
#   to learn which flags exist (--disk, --checkpointing, ...) and never passes
#   a flag the script doesn't declare. Checkpoint contract assumed (verified
#   against Splat_trainer2.py on github):
#       torch.save({"sd", "image_size", "num_packets"}, out/model2.pt)
#       recon_%06d.png / sample_%06d.png / loss.csv / faces_cache_{S}.npy in out/
#       log lines: "step N/M  rec R (PSNR P)  kl K  beta B  lr L  I img/s"
#
# HONESTY LEDGER (what was actually verified in the sandbox before shipping):
#   [V] app constructs offscreen; all four tabs build
#   [V] dataset prep: synthetic video -> Haar-face-cropped frames end to end
#   [V] training tab: launched real Splat_trainer2.py on 40 synthetic images
#       (32px/16 packets/60 steps) through the QProcess path; step/loss/PSNR
#       parsed live; recon preview picked up; process exit handled
#   [V] avatar tab: walk mode rendered frames from the checkpoint that training
#       run produced, through the same pursue()/render_image() as the driver
#   [ ] webcam mode needs a camera — wired identically to avatar_driver.py,
#       untested here by necessity
#   [ ] pulse check on an actual CUDA card — the math ran, but this sandbox is
#       CPU; the VRAM numbers on your 3060 are yours to see first
#   [ ] splat_trainer3v2.py itself — your upload did not land; adapter tested
#       against Splat_trainer2.py, flag detection is by source scan
# =============================================================================
import glob
import importlib.util
import math
import os
import re
import shutil
import sys
import time

import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from PyQt6.QtCore import (Qt, QThread, pyqtSignal, QTimer, QProcess,
                          QSize)
from PyQt6.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QFileDialog, QPlainTextEdit, QProgressBar,
    QGroupBox, QSlider, QMessageBox, QFormLayout, QFrame, QSizePolicy)

# ---------------------------------------------------------------- trainer adapter
TRAINER_CANDIDATES = ["splat_trainer3v2.py", "splat_trainer3.py",
                      "Splat_trainer3.py", "Splat_trainer2.py",
                      "splat_trainer2.py"]


def find_trainer():
    for name in TRAINER_CANDIDATES:
        p = os.path.join(APP_DIR, name)
        if os.path.exists(p):
            return p
    return None


def trainer_flags(path):
    """Which CLI flags does the trainer actually declare? (source scan)"""
    try:
        src = open(path, "r", encoding="utf-8", errors="replace").read()
    except OSError:
        return set()
    return set(re.findall(r'add_argument\(\s*"(--[\w-]+)"', src))


_TRAINER_MOD = None


def import_trainer(path):
    """Import the trainer module once (for SplatVAE etc. on the avatar tab)."""
    global _TRAINER_MOD
    if _TRAINER_MOD is None:
        spec = importlib.util.spec_from_file_location("splat_trainer_mod", path)
        _TRAINER_MOD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_TRAINER_MOD)
    return _TRAINER_MOD


# ---------------------------------------------------------------- certified math
# ported verbatim from avatar_driver.py — do not "improve"
def render_image(ren, P):
    import torch
    px, py, sigma, theta, freq, coeff = P
    out = None
    for i in range(0, ren.N, ren.chunk):
        sl = slice(i, i + ren.chunk)
        c = ren._chunk(px[:, sl], py[:, sl], sigma[:, sl],
                       theta[:, sl], freq[:, sl], coeff[:, sl])
        out = c if out is None else out + c
    return torch.sigmoid(out)


def _arc_step(a, b, alpha):
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + alpha * d


def pursue(P, T, alpha, mode):
    import torch
    px, py, s, th, f, c = P
    pxT, pyT, sT, thT, fT, cT = T
    L = lambda a, b: a + alpha * (b - a)
    px2, py2, s2, f2 = L(px, pxT), L(py, pyT), L(s, sT), L(f, fT)
    th2 = _arc_step(th, thT, alpha)
    if mode == "lerp":
        c2 = L(c, cT)
    else:
        a_, b_ = c[..., 0], c[..., 1]
        aT, bT = cT[..., 0], cT[..., 1]
        m = torch.sqrt(a_ * a_ + b_ * b_ + 1e-12)
        mT = torch.sqrt(aT * aT + bT * bT + 1e-12)
        ph = torch.atan2(b_, a_)
        phT = torch.atan2(bT, aT)
        m2 = L(m, mT)
        ph2 = _arc_step(ph, phT, alpha)
        c2 = torch.stack([m2 * torch.cos(ph2), m2 * torch.sin(ph2)], dim=-1)
    return (px2, py2, s2, th2, f2, c2)


def clone_params(P):
    return tuple(t.clone() for t in P)


def normalize_crop(x, tgt_mean=0.52, tgt_std=0.26):
    m, s = x.mean(), x.std() + 1e-6
    return np.clip((x - m) / s * tgt_std + tgt_mean, 0, 1)


# ---------------------------------------------------------------- theme
QSS = """
* { font-family: 'Segoe UI', 'Inter', sans-serif; }
QMainWindow, QWidget { background: #14161b; color: #d7dae0; }
QTabWidget::pane { border: 1px solid #262a33; border-radius: 6px; }
QTabBar::tab { background: #1a1d24; color: #8b91a0; padding: 9px 22px;
               border-top-left-radius: 6px; border-top-right-radius: 6px;
               margin-right: 2px; font-size: 13px; }
QTabBar::tab:selected { background: #232733; color: #e8b44c; font-weight: 600; }
QGroupBox { border: 1px solid #2a2f3a; border-radius: 8px; margin-top: 12px;
            padding-top: 16px; font-weight: 600; color: #a9b0bf; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
QPushButton { background: #2a3040; color: #e6e9ef; border: 1px solid #39415a;
              border-radius: 6px; padding: 7px 16px; font-size: 13px; }
QPushButton:hover { background: #353d52; }
QPushButton:pressed { background: #232838; }
QPushButton:disabled { background: #1c1f27; color: #565c69; }
QPushButton#accent { background: #b8862b; color: #14161b; font-weight: 700;
                     border: none; }
QPushButton#accent:hover { background: #d19c35; }
QPushButton#accent:disabled { background: #4a3d1e; color: #7a715c; }
QPushButton#danger { background: #7a2e2e; border: none; }
QPushButton#danger:hover { background: #944040; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #1b1e26; border: 1px solid #2c313d; border-radius: 5px;
    padding: 5px 8px; color: #d7dae0; }
QPlainTextEdit { background: #0e1013; border: 1px solid #262a33;
                 border-radius: 6px; color: #9fd08a;
                 font-family: 'Consolas', 'DejaVu Sans Mono', monospace;
                 font-size: 12px; }
QProgressBar { background: #1b1e26; border: 1px solid #2c313d;
               border-radius: 5px; text-align: center; color: #d7dae0; }
QProgressBar::chunk { background: #b8862b; border-radius: 4px; }
QSlider::groove:horizontal { height: 5px; background: #2c313d; border-radius: 2px; }
QSlider::handle:horizontal { width: 15px; margin: -6px 0; border-radius: 7px;
                             background: #e8b44c; }
QLabel#h1 { font-size: 30px; font-weight: 800; color: #e8b44c; }
QLabel#h2 { font-size: 15px; color: #a9b0bf; }
QLabel#stat { font-family: 'Consolas', monospace; font-size: 13px;
              color: #9fd08a; }
QLabel#warn { color: #d98e5f; }
QLabel#imgpane { background: #0e1013; border: 1px solid #262a33;
                 border-radius: 6px; }
QCheckBox { 
    spacing: 8px; 
}
QCheckBox::indicator {
    width: 18px; 
    height: 18px;
    background: #1b1e26;
    border: 1px solid #39415a;
    border-radius: 4px;
}
QCheckBox::indicator:hover {
    border: 1px solid #e8b44c;
}
QCheckBox::indicator:checked {
    background: #e8b44c;
    border: 1px solid #e8b44c;
    image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2314161b' stroke-width='4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='20 6 9 17 4 12'%3E%3C/polyline%3E%3C/svg%3E");
}
"""


def np_to_pixmap(arr, target=None):
    """HxWx3 uint8 RGB -> QPixmap (optionally scaled to fit target QSize)."""
    arr = np.ascontiguousarray(arr)
    h, w, _ = arr.shape
    im = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    pm = QPixmap.fromImage(im)
    if target is not None:
        pm = pm.scaled(target, Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
    return pm


# =============================================================================
# TAB 1 — HOME
# =============================================================================
class HomeTab(QWidget):
    def __init__(self, trainer_path):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(48, 40, 48, 40)
        lay.setSpacing(14)

        t = QLabel("TINY AVATAR"); t.setObjectName("h1")
        s = QLabel("a wave-interference face model, end to end")
        s.setObjectName("h2")
        lay.addWidget(t); lay.addWidget(s)
        lay.addSpacing(10)

        body = QLabel(
            "This studio wraps a ~7 MB generative model: a VAE that maps a "
            "128-dimensional latent to a few hundred Gabor wave packets, "
            "rendered by nothing but additive wave interference. No pixels "
            "stored, no convolutions in the decoder — the face IS the "
            "interference pattern.\n\n"
            "The avatar side runs on phase-transport pursuit: between "
            "encoder keyframes, packets glide along the complex-phasor "
            "geodesic instead of crossfading, which is what keeps the image "
            "from dissolving into 'fire' mid-motion. That mechanism passed a "
            "registered 8-pair gate (amplitude discipline, sharper-than-road "
            "mid-frames, scramble control broke 10x) before this app was "
            "built around it.\n\n"
            "Workflow:  Dataset Prep -> Training Studio -> Avatar Driver.\n"
            "Record one to two minutes of yourself talking and turning your "
            "head, extract face-cropped frames, train (this takes hours, not "
            "minutes — the studio shows you the model learning as it goes), "
            "then drive the result live from your webcam.")
        body.setWordWrap(True)
        body.setStyleSheet("font-size: 13.5px; line-height: 150%; color: #c3c8d2;")
        lay.addWidget(body)
        lay.addSpacing(8)

        tp = trainer_path or "NO TRAINER FOUND — put your trainer .py next to this app"
        eng = QLabel(f"engine: {os.path.basename(tp) if trainer_path else tp}")
        eng.setObjectName("stat" if trainer_path else "warn")
        lay.addWidget(eng)
        lay.addStretch(1)


# =============================================================================
# TAB 2 — DATASET PREP
# =============================================================================
class ExtractWorker(QThread):
    progress = pyqtSignal(int, int)          # done, total(-1 if unknown)
    preview = pyqtSignal(np.ndarray)
    finished_ok = pyqtSignal(int, str)       # n frames, out dir
    failed = pyqtSignal(str)

    def __init__(self, video, out, stride, size, use_face):
        super().__init__()
        self.video, self.out = video, out
        self.stride, self.size, self.use_face = stride, size, use_face
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            import cv2 as cv
            os.makedirs(self.out, exist_ok=True)
            cap = cv.VideoCapture(self.video)
            if not cap.isOpened():
                self.failed.emit(f"could not open {self.video}"); return
            # PHONE PORTRAIT TRAP: phones record sideways pixels plus a
            # rotation metadata tag. Every video player obeys the tag, so the
            # clip LOOKS upright — but cv2 hands you the raw sideways frames,
            # and the face detector only knows upright faces. We read the tag
            # and rotate deterministically ourselves (mapping verified
            # pixel-identical to OpenCV's CAP_PROP_ORIENTATION_AUTO):
            #   tag 90 -> ROTATE_90_CLOCKWISE, 180 -> 180, 270 -> CCW.
            # If your build predates these props, nothing rotates — the
            # preview will show it sideways; simplest fix: record LANDSCAPE.
            rot_op, meta = None, 0
            try:
                cap.set(cv.CAP_PROP_ORIENTATION_AUTO, 0)  # raw; we rotate
                meta = int(round(cap.get(cv.CAP_PROP_ORIENTATION_META))) % 360
                rot_op = {90: cv.ROTATE_90_CLOCKWISE,
                          180: cv.ROTATE_180,
                          270: cv.ROTATE_90_COUNTERCLOCKWISE}.get(meta)
            except Exception:
                pass
            if rot_op is not None:
                self.progress.emit(0, -1)   # touch UI early; work is starting
            total = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) or -1
            face = None
            if self.use_face:
                cpath = os.path.join(cv.data.haarcascades,
                                     "haarcascade_frontalface_default.xml")
                face = cv.CascadeClassifier(cpath)
                if face.empty():
                    face = None
            last_box = None
            i = n = 0
            while not self._stop:
                ok, fr = cap.read()
                if not ok:
                    break
                if rot_op is not None:
                    fr = cv.rotate(fr, rot_op)     # un-do the phone tag
                if i % self.stride == 0:
                    box = None
                    if face is not None:
                        g = cv.cvtColor(fr, cv.COLOR_BGR2GRAY)
                        det = face.detectMultiScale(g, 1.15, 5,
                                                    minSize=(80, 80))
                        if len(det):
                            x, y, w, h = max(det, key=lambda b: b[2] * b[3])
                            m = int(0.35 * max(w, h))   # margin: hair + chin
                            box = (max(x - m, 0), max(y - m, 0),
                                   min(x + w + m, fr.shape[1]),
                                   min(y + h + m, fr.shape[0]))
                            last_box = box
                        else:
                            box = last_box              # brief tracking dropout
                    if box is None:                     # fallback: center square
                        H, W = fr.shape[:2]; s = min(H, W)
                        box = ((W - s) // 2, (H - s) // 2,
                               (W + s) // 2, (H + s) // 2)
                    x0, y0, x1, y1 = box
                    # square it up around the center
                    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                    s = max(x1 - x0, y1 - y0) // 2
                    H, W = fr.shape[:2]
                    x0, x1 = max(cx - s, 0), min(cx + s, W)
                    y0, y1 = max(cy - s, 0), min(cy + s, H)
                    crop = fr[y0:y1, x0:x1]
                    if crop.size == 0:
                        i += 1; continue
                    crop = cv.resize(crop, (self.size, self.size),
                                     interpolation=cv.INTER_AREA)
                    cv.imwrite(os.path.join(self.out, f"f{n:05d}.jpg"), crop)
                    if n % 25 == 0:
                        self.preview.emit(crop[:, :, ::-1].copy())
                        self.progress.emit(i, total)
                    n += 1
                i += 1
            cap.release()
            if self._stop:
                self.failed.emit("stopped by user"); return
            if n == 0:
                self.failed.emit("no frames extracted"); return
            note = (f" (rotated {meta}\u00b0 from phone metadata)"
                    if rot_op is not None else "")
            self.finished_ok.emit(n, self.out + note)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class DatasetTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        lay = QVBoxLayout(self); lay.setContentsMargins(24, 20, 24, 20)

        note = QLabel(
            "One identity, good coverage: record 1-2 minutes of yourself "
            "TALKING, TURNING your head through angles, changing expression, "
            "with a little lighting variation. Coverage of pose + expression "
            "is what lets the single-identity manifold lock and track.")
        note.setWordWrap(True); note.setObjectName("h2")
        lay.addWidget(note)

        vg = QGroupBox("Video  ->  face-cropped frames")
        f = QFormLayout(vg)
        row = QHBoxLayout()
        self.video_edit = QLineEdit(); self.video_edit.setPlaceholderText(
            "path to your video (mp4/avi/mkv...)")
        b = QPushButton("Browse"); b.clicked.connect(self.pick_video)
        row.addWidget(self.video_edit); row.addWidget(b)
        f.addRow("video file", row)
        row2 = QHBoxLayout()
        self.out_edit = QLineEdit(os.path.join(APP_DIR, "faces1"))
        b2 = QPushButton("Browse"); b2.clicked.connect(self.pick_out)
        row2.addWidget(self.out_edit); row2.addWidget(b2)
        f.addRow("output folder", row2)
        self.stride = QSpinBox(); self.stride.setRange(1, 30); self.stride.setValue(2)
        f.addRow("keep every Nth frame", self.stride)
        self.size = QSpinBox(); self.size.setRange(64, 512); self.size.setValue(178)
        f.addRow("saved frame size (px)", self.size)
        self.face_chk = QCheckBox(
            "face-detect crop (recommended — otherwise your face is a blob "
            "in a wide frame)")
        self.face_chk.setChecked(True)
        f.addRow(self.face_chk)
        warn = QLabel("Extraction walks the whole video — on a long clip "
                      "this takes a while. Training afterwards takes HOURS. "
                      "Both are normal.\n"
                      "Phone videos: portrait clips are stored sideways with "
                      "a rotation tag — players hide this, extractors don't. "
                      "The app reads the tag and un-rotates automatically; "
                      "if the preview still shows you sideways, record in "
                      "LANDSCAPE and re-shoot.")
        warn.setObjectName("warn"); warn.setWordWrap(True)
        f.addRow(warn)
        hb = QHBoxLayout()
        self.go = QPushButton("Extract frames"); self.go.setObjectName("accent")
        self.go.clicked.connect(self.start)
        self.stop_btn = QPushButton("Stop"); self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self.stop)
        hb.addWidget(self.go); hb.addWidget(self.stop_btn); hb.addStretch(1)
        f.addRow(hb)
        self.bar = QProgressBar(); f.addRow(self.bar)
        lay.addWidget(vg)

        ig = QGroupBox("Already have images?")
        il = QHBoxLayout(ig)
        self.img_dir = QLineEdit(); self.img_dir.setPlaceholderText(
            "folder of jpg/png of ONE person — used directly by the trainer")
        b3 = QPushButton("Browse"); b3.clicked.connect(self.pick_imgdir)
        b4 = QPushButton("Check folder"); b4.clicked.connect(self.check_folder)
        il.addWidget(self.img_dir); il.addWidget(b3); il.addWidget(b4)
        lay.addWidget(ig)

        bottom = QHBoxLayout()
        self.preview = QLabel("frame preview"); self.preview.setObjectName("imgpane")
        self.preview.setFixedSize(QSize(260, 260))
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status = QLabel(""); self.status.setObjectName("stat")
        self.status.setWordWrap(True)
        bottom.addWidget(self.preview); bottom.addWidget(self.status, 1)
        lay.addLayout(bottom)
        lay.addStretch(1)

    # -- pickers
    def pick_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "Video", "",
                                           "Video (*.mp4 *.avi *.mkv *.mov *.webm);;All (*)")
        if p: self.video_edit.setText(p)

    def pick_out(self):
        p = QFileDialog.getExistingDirectory(self, "Output folder")
        if p: self.out_edit.setText(p)

    def pick_imgdir(self):
        p = QFileDialog.getExistingDirectory(self, "Image folder")
        if p: self.img_dir.setText(p)

    def check_folder(self):
        d = self.img_dir.text().strip()
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        n = sum(len(glob.glob(os.path.join(d, e))) for e in exts)
        self.status.setText(
            f"{n} images in {d}\n" +
            ("Fine to train on directly — set this as data_dir in the "
             "Training tab." if n >= 200 else
             "Under ~200 images is thin for pose+expression coverage; "
             "the model will still train but tracking range will be narrow."))

    # -- extraction
    def start(self):
        v = self.video_edit.text().strip()
        if not os.path.exists(v):
            QMessageBox.warning(self, "Tiny Avatar", "Video file not found."); return
        self.worker = ExtractWorker(v, self.out_edit.text().strip(),
                                    self.stride.value(), self.size.value(),
                                    self.face_chk.isChecked())
        self.worker.progress.connect(self.on_prog)
        self.worker.preview.connect(self.on_prev)
        self.worker.finished_ok.connect(self.on_done)
        self.worker.failed.connect(self.on_fail)
        self.go.setEnabled(False); self.stop_btn.setEnabled(True)
        self.status.setText("extracting...")
        self.worker.start()

    def stop(self):
        if self.worker: self.worker.stop()

    def on_prog(self, done, total):
        if total > 0:
            self.bar.setMaximum(total); self.bar.setValue(done)

    def on_prev(self, rgb):
        self.preview.setPixmap(np_to_pixmap(rgb, self.preview.size()))

    def on_done(self, n, out):
        self.go.setEnabled(True); self.stop_btn.setEnabled(False)
        self.bar.setValue(self.bar.maximum())
        self.status.setText(
            f"wrote {n} frames -> {out}\nNext: Training Studio tab, set "
            f"data_dir to this folder. First run builds a one-time .npy cache.")

    def on_fail(self, msg):
        self.go.setEnabled(True); self.stop_btn.setEnabled(False)
        self.status.setText(f"extraction failed: {msg}")


# =============================================================================
# TAB 3 — TRAINING STUDIO
# =============================================================================
LOG_RE = re.compile(
    r"step\s+(\d+)\s*/\s*(\d+)\s+rec\s+([\d.eE+-]+)\s+\(PSNR\s+([\d.]+)\)"
    r"\s+kl\s+([\d.eE+-]+)")


class TrainTab(QWidget):
    def __init__(self, trainer_path, flags):
        super().__init__()
        self.trainer_path, self.flags = trainer_path, flags
        self.proc = None
        self.last_shown = ""

        outer = QHBoxLayout(self); outer.setContentsMargins(20, 16, 20, 16)
        left = QVBoxLayout(); right = QVBoxLayout()
        outer.addLayout(left, 0); outer.addLayout(right, 1)

        # ---- settings form
        sg = QGroupBox("Run settings"); f = QFormLayout(sg)
        row = QHBoxLayout()
        self.data_dir = QLineEdit(os.path.join(APP_DIR, "faces1"))
        b = QPushButton("..."); b.setFixedWidth(30); b.clicked.connect(self.pick_data)
        row.addWidget(self.data_dir); row.addWidget(b)
        f.addRow("data_dir", row)
        row2 = QHBoxLayout()
        self.out_dir = QLineEdit(os.path.join(APP_DIR, "runs", "tiny1"))
        b2 = QPushButton("..."); b2.setFixedWidth(30); b2.clicked.connect(self.pick_out)
        row2.addWidget(self.out_dir); row2.addWidget(b2)
        f.addRow("out dir", row2)
        self.res = QComboBox(); self.res.addItems(["64", "96", "128", "160", "192"])
        self.res.setCurrentText("128")
        f.addRow("image_size", self.res)
        self.packets = QSpinBox(); self.packets.setRange(32, 2048)
        self.packets.setSingleStep(64); self.packets.setValue(512)
        f.addRow("num_packets", self.packets)
        self.batch = QSpinBox(); self.batch.setRange(4, 512); self.batch.setValue(64)
        f.addRow("batch", self.batch)
        self.steps = QSpinBox(); self.steps.setRange(100, 500000)
        self.steps.setSingleStep(1000); self.steps.setValue(30000)
        f.addRow("steps", self.steps)
        self.beta = QDoubleSpinBox(); self.beta.setDecimals(5)
        self.beta.setRange(0.0, 10.0); self.beta.setValue(0.001)
        f.addRow("beta (low = sharp single identity)", self.beta)
        self.lr = QDoubleSpinBox(); self.lr.setDecimals(6)
        self.lr.setRange(1e-6, 1e-1); self.lr.setValue(3e-4)
        f.addRow("lr", self.lr)
        self.gamma = QDoubleSpinBox(); self.gamma.setDecimals(4)
        self.gamma.setRange(0.0, 1.0); self.gamma.setValue(0.02)
        f.addRow("gamma_floater (0 = off)", self.gamma)
        self.log_every = QSpinBox(); self.log_every.setRange(10, 5000)
        self.log_every.setValue(250)
        f.addRow("log/save every N steps", self.log_every)
        self.disk_chk = QCheckBox("--disk cache fallback")
        self.disk_chk.setEnabled("--disk" in self.flags)
        if "--disk" not in self.flags:
            self.disk_chk.setText("--disk (not declared by this trainer)")
        f.addRow(self.disk_chk)
        self.ckpt_chk = QCheckBox("--checkpointing (trade speed for VRAM)")
        self.ckpt_chk.setEnabled("--checkpointing" in self.flags)
        f.addRow(self.ckpt_chk)
        left.addWidget(sg)

        # ---- pulse
        pg = QGroupBox("Pulse check"); pl = QVBoxLayout(pg)
        self.pulse_btn = QPushButton("Take the pulse")
        self.pulse_btn.clicked.connect(self.pulse)
        self.pulse_lbl = QLabel("—"); self.pulse_lbl.setObjectName("stat")
        self.pulse_lbl.setWordWrap(True)
        pl.addWidget(self.pulse_btn); pl.addWidget(self.pulse_lbl)
        left.addWidget(pg)

        # ---- controls
        cg = QGroupBox("Run"); cl = QVBoxLayout(cg)
        self.resume_lbl = QLabel(""); self.resume_lbl.setObjectName("warn")
        self.resume_lbl.setWordWrap(True)
        cl.addWidget(self.resume_lbl)
        hb = QHBoxLayout()
        self.start_btn = QPushButton("Start training")
        self.start_btn.setObjectName("accent")
        self.start_btn.clicked.connect(lambda: self.start(resume=False))
        self.resume_btn = QPushButton("Resume")
        self.resume_btn.clicked.connect(lambda: self.start(resume=True))
        self.stop_btn = QPushButton("Stop"); self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self.stop)
        hb.addWidget(self.start_btn); hb.addWidget(self.resume_btn)
        hb.addWidget(self.stop_btn)
        cl.addLayout(hb)
        self.prog = QProgressBar(); cl.addWidget(self.prog)
        self.stat_lbl = QLabel("idle"); self.stat_lbl.setObjectName("stat")
        self.stat_lbl.setWordWrap(True)
        cl.addWidget(self.stat_lbl)
        self.sys_lbl = QLabel(""); self.sys_lbl.setObjectName("stat")
        cl.addWidget(self.sys_lbl)
        left.addWidget(cg)
        left.addStretch(1)

        # ---- right side: previews + console
        prow = QHBoxLayout()
        self.recon_lbl = QLabel("recon preview\n(appears at first log step)")
        self.sample_lbl = QLabel("sample preview")
        for w in (self.recon_lbl, self.sample_lbl):
            w.setObjectName("imgpane")
            w.setMinimumSize(QSize(300, 300))
            w.setAlignment(Qt.AlignmentFlag.AlignCenter)
            w.setSizePolicy(QSizePolicy.Policy.Expanding,
                            QSizePolicy.Policy.Expanding)
            prow.addWidget(w)
        right.addLayout(prow, 1)
        self.console = QPlainTextEdit(); self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(2000)
        right.addWidget(self.console, 1)

        # timers
        self.sys_timer = QTimer(self); self.sys_timer.timeout.connect(self.sys_tick)
        self.sys_timer.start(1500)
        self.img_timer = QTimer(self); self.img_timer.timeout.connect(self.img_tick)
        self.out_dir.textChanged.connect(self.scan_resume)
        self.scan_resume()

    # -- pickers
    def pick_data(self):
        p = QFileDialog.getExistingDirectory(self, "Data dir")
        if p: self.data_dir.setText(p)

    def pick_out(self):
        p = QFileDialog.getExistingDirectory(self, "Out dir")
        if p: self.out_dir.setText(p)

    # -- resume detection
    def scan_resume(self):
        out = self.out_dir.text().strip()
        pt = os.path.join(out, "model2.pt")
        caches = glob.glob(os.path.join(out, "faces_cache_*.npy"))
        bits = []
        if os.path.exists(pt):
            age = (time.time() - os.path.getmtime(pt)) / 3600
            bits.append(f"checkpoint found ({age:.1f} h old) — Resume continues it")
        if caches:
            bits.append(f"cache present ({os.path.basename(caches[0])}) — "
                        "no re-preprocessing needed")
        self.resume_lbl.setText("\n".join(bits) if bits
                                else "fresh run — no checkpoint in this out dir")
        self.resume_btn.setEnabled(os.path.exists(pt))

    # -- pulse
    def pulse(self):
        import torch
        S = int(self.res.currentText())
        B, N = self.batch.value(), self.packets.value()
        chunk = 64
        lines = []
        # dataset side
        d = self.data_dir.text().strip()
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        n_img = sum(len(glob.glob(os.path.join(d, e))) for e in exts)
        cache_gb = n_img * 3 * S * S / 1e9
        lines.append(f"dataset: {n_img} images -> cache {cache_gb:.2f} GB "
                     f"(uint8 {S}px)")
        # renderer working-set heuristic: per-chunk trig + envelope maps,
        # fp32, x~6 intermediates. Heuristic, not a promise.
        work_gb = B * chunk * S * S * 4 * 6 / 1e9
        lines.append(f"renderer working set ~{work_gb:.2f} GB "
                     f"(batch {B} x chunk {chunk} @ {S}px, heuristic x6)")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            lines.append(f"VRAM: {free/1e9:.2f} GB free / {total/1e9:.2f} GB total")
            verdict = []
            if cache_gb > free / 1e9 - 3.0:
                verdict.append("dataset will NOT sit in VRAM (trainer keeps "
                               "3 GB headroom)")
                if "--disk" in self.flags:
                    self.disk_chk.setChecked(True)
                    verdict.append("-> --disk engaged automatically")
                else:
                    verdict.append("-> this trainer has no --disk flag; it "
                                   "will pin to RAM instead")
            else:
                verdict.append("dataset fits resident in VRAM")
            if work_gb > free / 1e9 * 0.5:
                verdict.append(f"renderer estimate is over half your free "
                               f"VRAM — consider batch {max(8, B//2)} or "
                               "--checkpointing")
            else:
                verdict.append("renderer estimate looks comfortable")
            lines += verdict
        else:
            lines.append("no CUDA visible from here — CPU training works but "
                         "is 10-100x slower; the numbers above still apply "
                         "to RAM")
        self.pulse_lbl.setText("\n".join(lines))

    # -- run control
    def start(self, resume=False):
        if self.proc is not None:
            return
        if not self.trainer_path:
            QMessageBox.critical(self, "Tiny Avatar",
                                 "No trainer script found next to the app.")
            return
        out = self.out_dir.text().strip()
        os.makedirs(out, exist_ok=True)
        args = ["-u", self.trainer_path,
                "--data_dir", self.data_dir.text().strip(),
                "--out", out,
                "--image_size", self.res.currentText(),
                "--num_packets", str(self.packets.value()),
                "--batch", str(self.batch.value()),
                "--steps", str(self.steps.value()),
                "--beta", f"{self.beta.value():g}",
                "--lr", f"{self.lr.value():g}"]
        if "--gamma_floater" in self.flags:
            args += ["--gamma_floater", f"{self.gamma.value():g}"]
        if "--log_every" in self.flags:
            args += ["--log_every", str(self.log_every.value())]
        if resume:
            pt = os.path.join(out, "model2.pt")
            if os.path.exists(pt) and "--resume" in self.flags:
                args += ["--resume", pt]
        if self.disk_chk.isChecked() and "--disk" in self.flags:
            args += ["--disk"]
        if self.ckpt_chk.isChecked() and "--checkpointing" in self.flags:
            args += ["--checkpointing"]

        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.on_out)
        self.proc.finished.connect(self.on_fin)
        self.console.appendPlainText(
            f"$ {sys.executable} {' '.join(args)}\n")
        self.proc.start(sys.executable, args)
        self.start_btn.setEnabled(False); self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.stat_lbl.setText("launching trainer process...")
        self.prog.setMaximum(self.steps.value()); self.prog.setValue(0)
        self.img_timer.start(2000)

    def stop(self):
        if self.proc:
            self.console.appendPlainText("\n[stopping trainer — checkpoint "
                                         "is saved every log step, Resume "
                                         "will pick it up]")
            self.proc.kill()

    def on_out(self):
        txt = bytes(self.proc.readAllStandardOutput()).decode(
            "utf-8", errors="replace")
        for line in txt.splitlines():
            if line.strip():
                self.console.appendPlainText(line)
            m = LOG_RE.search(line)
            if m:
                step, tot = int(m.group(1)), int(m.group(2))
                rec, psnr, klv = m.group(3), m.group(4), m.group(5)
                self.prog.setMaximum(tot); self.prog.setValue(step)
                self.stat_lbl.setText(
                    f"step {step}/{tot}   rec {rec}   PSNR {psnr} dB   "
                    f"kl {klv}")

    def on_fin(self, code, _status):
        self.console.appendPlainText(f"\n[trainer exited, code {code}]")
        self.proc = None
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.img_timer.stop()
        self.img_tick()
        self.scan_resume()
        self.stat_lbl.setText(f"stopped (exit {code})" if code else "done")

    # -- system + preview ticks
    def sys_tick(self):
        bits = []
        try:
            import psutil
            vm = psutil.virtual_memory()
            bits.append(f"RAM {vm.used/1e9:.1f}/{vm.total/1e9:.1f} GB "
                        f"({vm.percent:.0f}%)  CPU {psutil.cpu_percent():.0f}%")
        except Exception:
            pass
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            mi = pynvml.nvmlDeviceGetMemoryInfo(h)
            bits.append(f"GPU {u.gpu}%  VRAM {mi.used/1e9:.1f}/"
                        f"{mi.total/1e9:.1f} GB")
        except Exception:
            try:
                import torch
                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info()
                    bits.append(f"VRAM {(total-free)/1e9:.1f}/"
                                f"{total/1e9:.1f} GB used")
            except Exception:
                pass
        self.sys_lbl.setText("   ".join(bits))

    def img_tick(self):
        out = self.out_dir.text().strip()
        for pat, lbl in (("recon_*.png", self.recon_lbl),
                         ("sample_*.png", self.sample_lbl)):
            files = sorted(glob.glob(os.path.join(out, pat)))
            if files and files[-1] != getattr(lbl, "_shown", None):
                pm = QPixmap(files[-1])
                if not pm.isNull():
                    lbl.setPixmap(pm.scaled(
                        lbl.size(), Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation))
                    lbl._shown = files[-1]


# =============================================================================
# TAB 4 — AVATAR DRIVER
# =============================================================================
class AvatarWorker(QThread):
    frame = pyqtSignal(np.ndarray, np.ndarray, float)   # cam_rgb, avatar_rgb, fps
    status = pyqtSignal(str)

    def __init__(self, trainer_path, model_path, source):
        super().__init__()
        self.trainer_path, self.model_path = trainer_path, model_path
        self.source = source            # "webcam" | "walk"
        self.mode = "phase"             # direct | lerp | phase
        self.kf = 8
        self.alpha = 0.35
        self.norm = True
        self.walk_step, self.z_max = 2.5, 12.0
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            import torch
            ST = import_trainer(self.trainer_path)
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ck = torch.load(self.model_path, map_location="cpu")
            model = ST.SplatVAE(ck["image_size"], ck["num_packets"])
            model.load_state_dict(ck["sd"]); model.eval().to(dev)
            ren = model.ren
            self.status.emit(f"model {ck['image_size']}px / "
                             f"{ck['num_packets']} packets on {dev}")
            if self.source == "webcam":
                self._webcam(model, ren, dev, torch)
            else:
                self._walk(model, ren, dev, torch)
        except Exception as e:
            self.status.emit(f"avatar failed: {type(e).__name__}: {e}")

    # ---- webcam loop (wiring identical to avatar_driver.py live mode)
    def _webcam(self, model, ren, dev, torch):
        import cv2 as cv
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            self.status.emit("webcam failed to open — try Latent walk instead")
            return
        P = T = None
        f = 0
        t_last, fps = time.time(), 0.0
        while not self._stop:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]; s = min(h, w)
            crop = frame[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
            x = cv.resize(crop, (ren.H, ren.W))[:, :, ::-1].astype(
                np.float32) / 255.0
            if self.norm:
                x = normalize_crop(x)
            xt = torch.from_numpy(np.ascontiguousarray(
                x.transpose(2, 0, 1)))[None].to(dev)
            need = (self.mode == "direct") or (f % max(1, self.kf) == 0) \
                or (T is None)
            if need:
                with torch.no_grad():
                    mu, _ = model.enc(xt)
                    T = ren.activate(model.dec(mu).float())
                if P is None:
                    P = clone_params(T)
            with torch.no_grad():
                if self.mode == "direct":
                    P = clone_params(T)
                else:
                    P = pursue(P, T, self.alpha, self.mode)
                img = render_image(ren, P)
            av = (img[0].clamp(0, 1) * 255).byte().permute(
                1, 2, 0).cpu().numpy()
            cam = cv.resize(crop, (256, 256))[:, :, ::-1].copy()
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6); t_last = now
            self.frame.emit(cam, av, fps)
            f += 1
        cap.release()

    # ---- latent walk (no camera)
    def _walk(self, model, ren, dev, torch):
        LATENT = getattr(import_trainer(self.trainer_path), "LATENT", 128)
        g = torch.Generator().manual_seed(0)
        z = torch.randn(1, LATENT, generator=g).to(dev)

        def keyframe(z):
            with torch.no_grad():
                return ren.activate(model.dec(z).float())

        T = keyframe(z); P = clone_params(T)
        f = 0
        t_last, fps = time.time(), 0.0
        blank = np.zeros((256, 256, 3), np.uint8)
        while not self._stop:
            if f % max(1, self.kf) == 0:
                step = torch.randn(1, LATENT, generator=g).to(dev)
                z = z + self.walk_step * step
                z = z * min(1.0, self.z_max / (z.norm() + 1e-9))
                T = keyframe(z)
            with torch.no_grad():
                mode = self.mode if self.mode != "direct" else "phase"
                P = pursue(P, T, self.alpha, mode)
                img = render_image(ren, P)
            av = (img[0].clamp(0, 1) * 255).byte().permute(
                1, 2, 0).cpu().numpy()
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6); t_last = now
            self.frame.emit(blank, av, fps)
            f += 1
            time.sleep(max(0.0, 1 / 30 - (time.time() - now)))


class AvatarTab(QWidget):
    def __init__(self, trainer_path):
        super().__init__()
        self.trainer_path = trainer_path
        self.worker = None
        lay = QVBoxLayout(self); lay.setContentsMargins(20, 16, 20, 16)

        top = QHBoxLayout()
        self.model_combo = QComboBox(); self.refresh_models()
        rb = QPushButton("Rescan"); rb.clicked.connect(self.refresh_models)
        mb = QPushButton("Browse..."); mb.clicked.connect(self.pick_model)
        top.addWidget(QLabel("model")); top.addWidget(self.model_combo, 1)
        top.addWidget(rb); top.addWidget(mb)
        lay.addLayout(top)

        panes = QHBoxLayout()
        self.cam_lbl = QLabel("camera"); self.av_lbl = QLabel("avatar")
        for w in (self.cam_lbl, self.av_lbl):
            w.setObjectName("imgpane")
            w.setMinimumSize(QSize(360, 360))
            w.setAlignment(Qt.AlignmentFlag.AlignCenter)
            w.setSizePolicy(QSizePolicy.Policy.Expanding,
                            QSizePolicy.Policy.Expanding)
            panes.addWidget(w)
        lay.addLayout(panes, 1)

        ctl = QGroupBox("Drive"); g = QGridLayout(ctl)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["phase pursuit (certified transport)",
                                  "lerp pursuit (baseline)",
                                  "direct (encode every frame)"])
        self.mode_combo.currentIndexChanged.connect(self.push_params)
        g.addWidget(QLabel("mode"), 0, 0); g.addWidget(self.mode_combo, 0, 1)
        self.kf_sl = QSlider(Qt.Orientation.Horizontal)
        self.kf_sl.setRange(1, 30); self.kf_sl.setValue(8)
        self.kf_val = QLabel("8")
        self.kf_sl.valueChanged.connect(
            lambda v: (self.kf_val.setText(str(v)), self.push_params()))
        g.addWidget(QLabel("keyframe every N frames"), 1, 0)
        g.addWidget(self.kf_sl, 1, 1); g.addWidget(self.kf_val, 1, 2)
        self.al_sl = QSlider(Qt.Orientation.Horizontal)
        self.al_sl.setRange(5, 100); self.al_sl.setValue(35)
        self.al_val = QLabel("0.35")
        self.al_sl.valueChanged.connect(
            lambda v: (self.al_val.setText(f"{v/100:.2f}"), self.push_params()))
        g.addWidget(QLabel("pursuit alpha"), 2, 0)
        g.addWidget(self.al_sl, 2, 1); g.addWidget(self.al_val, 2, 2)
        self.norm_chk = QCheckBox("normalize input (fights the dark-head "
                                  "domain gap)")
        self.norm_chk.setChecked(True)
        self.norm_chk.toggled.connect(self.push_params)
        g.addWidget(self.norm_chk, 3, 0, 1, 3)
        hb = QHBoxLayout()
        self.cam_btn = QPushButton("Start webcam"); self.cam_btn.setObjectName("accent")
        self.cam_btn.clicked.connect(lambda: self.start("webcam"))
        self.walk_btn = QPushButton("Latent walk (no camera)")
        self.walk_btn.clicked.connect(lambda: self.start("walk"))
        self.stop_btn = QPushButton("Stop"); self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self.stop)
        hb.addWidget(self.cam_btn); hb.addWidget(self.walk_btn)
        hb.addWidget(self.stop_btn); hb.addStretch(1)
        g.addLayout(hb, 4, 0, 1, 3)
        lay.addWidget(ctl)
        self.status = QLabel("load a model, then Start"); self.status.setObjectName("stat")
        lay.addWidget(self.status)

    def refresh_models(self):
        self.model_combo.clear()
        pats = [os.path.join(APP_DIR, "runs", "*", "*.pt"),
                os.path.join(APP_DIR, "runs", "*", "*", "*.pt"),
                os.path.join(APP_DIR, "*.pt")]
        seen = []
        for p in pats:
            for f in sorted(glob.glob(p)):
                if f not in seen:
                    seen.append(f)
        for f in seen:
            self.model_combo.addItem(os.path.relpath(f, APP_DIR), f)
        if not seen:
            self.model_combo.addItem("(no .pt found — train one, or Browse)", "")

    def pick_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Model", APP_DIR,
                                           "PyTorch (*.pt)")
        if p:
            self.model_combo.insertItem(0, os.path.basename(p), p)
            self.model_combo.setCurrentIndex(0)

    def _mode(self):
        return ["phase", "lerp", "direct"][self.mode_combo.currentIndex()]

    def push_params(self):
        if self.worker:
            self.worker.mode = self._mode()
            self.worker.kf = self.kf_sl.value()
            self.worker.alpha = self.al_sl.value() / 100
            self.worker.norm = self.norm_chk.isChecked()

    def start(self, source):
        mp = self.model_combo.currentData()
        if not mp or not os.path.exists(mp):
            QMessageBox.warning(self, "Tiny Avatar",
                                "Pick a trained .pt model first "
                                "(Training Studio produces model2.pt).")
            return
        self.stop()
        self.worker = AvatarWorker(self.trainer_path, mp, source)
        self.push_params()
        self.worker.frame.connect(self.on_frame)
        self.worker.status.connect(self.status.setText)
        self.worker.start()
        self.cam_btn.setEnabled(False); self.walk_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop(self):
        if self.worker:
            self.worker.stop(); self.worker.wait(2000)
            self.worker = None
        self.cam_btn.setEnabled(True); self.walk_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_frame(self, cam, av, fps):
        self.cam_lbl.setPixmap(np_to_pixmap(cam, self.cam_lbl.size()))
        self.av_lbl.setPixmap(np_to_pixmap(av, self.av_lbl.size()))
        self.status.setText(
            f"{self._mode()}  kf {self.kf_sl.value()}  "
            f"alpha {self.al_sl.value()/100:.2f}  {fps:.0f} fps")


# =============================================================================
# main window
# =============================================================================
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tiny Avatar")
        self.resize(1180, 760)
        trainer = find_trainer()
        flags = trainer_flags(trainer) if trainer else set()
        tabs = QTabWidget()
        tabs.addTab(HomeTab(trainer), "Home")
        tabs.addTab(DatasetTab(), "Dataset Prep")
        self.train_tab = TrainTab(trainer, flags)
        tabs.addTab(self.train_tab, "Training Studio")
        self.avatar_tab = AvatarTab(trainer)
        tabs.addTab(self.avatar_tab, "Avatar Driver")
        self.setCentralWidget(tabs)

    def closeEvent(self, ev):
        try:
            self.avatar_tab.stop()
            if self.train_tab.proc:
                self.train_tab.proc.kill()
        except Exception:
            pass
        ev.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    w = Main()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()