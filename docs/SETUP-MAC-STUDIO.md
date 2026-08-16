# Running this on the Mac Studio

Written for Apple Silicon. The pipeline is GPU-bound and Demucs uses Metal (MPS),
so the Studio does in roughly real time what the Air would take several minutes
per song to grind through.

## 1. Prerequisites

macOS ships Python 3.9 at best. There is no system Python new enough for this,
so this step is required, not a convenience.

```sh
brew install python@3.12 node ffmpeg
```

**Use 3.12.** Not 3.11 (fine, but Homebrew may no longer keg it) and
specifically **not 3.13**, which is Homebrew's current default `python3` —
Demucs 4.x predates it and has no wheels, so `pip install -e '.[audio]'` will
try to build from source and fail.

`ffmpeg` is not optional either — librosa and Demucs both shell out to it for
anything that is not a WAV.

If `brew` itself is missing, install Homebrew first from https://brew.sh.

## 2. Clone and install

```sh
git clone https://github.com/4list233/DongCiDaCi.git
cd DongCiDaCi

python3.12 -m venv .venv
source .venv/bin/activate       # everything below assumes this is active
```

Note that `pip` and `dcdc` only exist *inside* the venv. If either reports
"command not found", the venv was never created or never activated — fix that
before anything else.

```sh
pip install -e .            # API, chart model, tests — seconds, no ML
pip install -e '.[audio]'   # the pipeline: torch, demucs, librosa — ~2GB
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

The real transcription model, and the reason the spectral fallback exists at
all. This is the bigger of the two upgrades.

Use **ADTOF-pytorch**, a port that needs only torch, librosa and pretty_midi:

```sh
git clone https://github.com/xavriley/ADTOF-pytorch
pip install -e ADTOF-pytorch
```

Same model to within ~0.2% F-measure, and critically it needs neither
TensorFlow nor madmom — those two are exactly what does not build on Apple
Silicon (madmom wants a Cython build against an old NumPy ABI, which is also
why Omnizart is unusable on ARM). Upstream ADTOF needs Python 3.10 plus both of
them, so it is not worth the fight on this machine.

Two things it buys you beyond accuracy:

- **Velocities.** It emits MIDI, so hits carry velocity, and the quantizer can
  actually distinguish a ghost note from a backbeat. The spectral fallback
  estimates velocity from band energy, which is far cruder.
- **Toms and cymbals.** The fallback has no tom or cymbal classes at all.

It still resolves only 5 classes, so ride-vs-crash and open-vs-closed hi-hat
remain yours to mark.

Note the underlying ADTOF work is **CC BY-NC-SA** — fine for a personal hobby,
not for anything commercial. Demucs, librosa and Beat This! are permissive.

Once `dcdc doctor` reports `adtof ok` the pipeline picks it up automatically;
`transcribe.py` selects the best available backend on its own.

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

**`zsh: command not found: python3.11`** — the Homebrew Python in step 1 was
never installed, or was installed under a different version. Check what you
actually have with `ls /opt/homebrew/bin/python3.*` and use that version in the
`venv` command.

**`zsh: command not found: pip` or `: dcdc`** — both live inside the venv. If
the `python3.x -m venv` line failed, nothing after it ran against a venv, and
these will be missing. Re-run from step 2.

**`pip install '.[audio]'` tries to build Demucs from source and fails** —
you are on Python 3.13. Delete `.venv`, recreate it with `python3.12`, and
reinstall.

**`npm warn allow-scripts ... fsevents`** — newer npm defers package install
scripts. Builds work without it; only the Vite dev server is affected, and it
falls back to polling for file changes. Run `npm approve-scripts --allow-scripts-pending`
in `frontend/` if you want native file watching back.

**`demucs failed (exit 1)`** — usually ffmpeg. Check `ffmpeg -version`.

**Charts come out displaced by a beat.** The downbeat phase was inferred wrong.
Install Beat This!; if it still happens, the grid needs shifting — `Grid.shift_downbeats()`
exists for this and is not yet wired to a button.

**Blank notation, no error.** An articulation name alphaTab does not recognise
fails the entire score rather than one note. Run `npm --prefix frontend test`,
which parses generated AlphaTex through alphaTab and checks every lane.

**`no space left on device`** — Demucs stems are large. `rm -rf songs/*/stems/`;
they regenerate.
