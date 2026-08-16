# Running this on the Mac Studio

Written for Apple Silicon. The pipeline is GPU-bound and Demucs uses Metal (MPS),
so the Studio does in roughly real time what the Air would take several minutes
per song to grind through.

## 1. Prerequisites

```sh
brew install python@3.11 node ffmpeg
```

`ffmpeg` is not optional — librosa and Demucs both shell out to it for anything
that is not a WAV.

## 2. Clone and install

```sh
git clone https://github.com/4list233/DongCiDaCi.git
cd DongCiDaCi

python3.11 -m venv .venv
source .venv/bin/activate

pip install -e .            # API, chart model, tests — fast, no ML
pip install -e '.[audio]'   # the pipeline: torch, demucs, librosa — slow, large
```

Then the frontend:

```sh
cd frontend && npm install && npm run build && cd ..
```

## 3. Check what you got

```sh
dcdc doctor
```

```
device            mps
demucs            ok        stage 1, drum isolation
beat_this         MISSING   stage 2, preferred beat tracker
librosa           ok        stage 2/3 fallback
adtof             MISSING   stage 3, real ADT model
```

`device mps` is the one that matters — that is Metal, and it means Demucs is
using the GPU. If it says `cpu`, torch installed without MPS support; reinstall
torch from the default PyPI wheel (the CPU-only index will not give you Metal).

## 4. Run it

```sh
dcdc serve                  # http://127.0.0.1:8000
```

For frontend work, run the Vite dev server alongside it and use port 5173 —
it proxies `/api` through to uvicorn:

```sh
npm --prefix frontend run dev
```

---

## Upgrading the two missing stages

The app runs without either of these. It just runs *worse*, and says so in the
banner rather than letting you believe otherwise.

### Beat This! — stage 2

Straight win, no dependency pain. Replaces librosa's naive downbeat inference,
which is the difference between a chart that starts on beat 1 and one that is
displaced by an eighth for its entire length.

```sh
pip install 'git+https://github.com/CPJKU/beat_this.git'
```

### ADTOF — stage 3

This is the real transcription model, and the reason the spectral fallback
exists at all. Two routes, and on Apple Silicon **only the first is realistic**:

**ADTOF-pytorch (recommended).** Same model, ~0.2% F-measure difference, and
critically it needs neither TensorFlow nor madmom. Those two are exactly what
breaks on ARM — madmom needs a Cython build against an old NumPy ABI, and
Omnizart is unusable on Apple Silicon for the same reason.

**Upstream ADTOF.** Needs Python 3.10, TensorFlow, Keras and madmom. If you want
it, give it its own environment rather than fighting the main one:

```sh
python3.10 -m venv .venv-adtof
source .venv-adtof/bin/activate
pip install 'git+https://github.com/MZehren/ADTOF.git'
```

Note ADTOF is **CC BY-NC-SA** — fine for a personal hobby, not for anything
commercial. Demucs, librosa and Beat This! are permissive.

Either way, once `dcdc doctor` reports `adtof ok`, the pipeline picks it up
automatically — `transcribe.py` chooses the best available backend and falls
back on its own.

---

## What "good" looks like

From `DrumSheet`'s published numbers, a competent classifier lands around
**87.8% detection with 1.95% false positives**. So expect roughly one hit in
eight to need your ear even when everything is installed correctly. That is not
a bug in the setup — it is the reason the editor exists.

The spectral fallback is well below that. It finds kick, snare and hi-hat only,
and it will not find ghost notes.

---

## Where the time goes

Rough shape on an M-series Studio, for a four-minute song:

| Stage | Time | Notes |
|-------|------|-------|
| Demucs separation | 20–60s | The bulk of it. `htdemucs_ft` is ~4× slower for a small gain |
| Beat tracking | 5–15s | |
| Transcription | 5–20s | |
| Quantization | <1s | Pure Python, no model |

One job runs at a time on purpose. The pipeline is GPU-bound, so running two
concurrently would only make both slower.

---

## Moving work between machines

The repo *is* the migration format. Charts, notes, and the raw machine baseline
are committed; audio and stems are gitignored. So:

```sh
git pull        # on the Studio
```

gets you every chart from the Air, and the audio files stay where they are. If
you want a song's audio on both machines, copy `songs/<slug>/source.*` by hand —
deliberately not in git, since it is neither yours to redistribute nor small.

## Troubleshooting

**`demucs failed (exit 1)`** — usually ffmpeg. Check `ffmpeg -version`.

**Charts come out displaced by a beat.** The downbeat phase was inferred wrong.
Install Beat This!; if it still happens, the grid needs shifting — `Grid.shift_downbeats()`
exists for this and is not yet wired to a button.

**Blank notation, no error.** An articulation name alphaTab does not recognise
fails the entire score rather than one note. Run `npm --prefix frontend test`,
which parses generated AlphaTex through alphaTab and checks every lane.

**`no space left on device`** — Demucs stems are large. `rm -rf songs/*/stems/`;
they regenerate.
