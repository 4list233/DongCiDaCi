# How to actually get accurate drum transcription

Research notes behind the stage 1b/3 rebuild, August 2026.

## The finding

**Classifying a mixed drum track is the wrong problem.** Every 5-class model —
ADTOF included — is being asked "which of these overlapping things did I just
hear", on audio where a crash, a ride, and a hi-hat share most of their
spectrum. That is where toms vanish, ride collapses into crash, and ghost notes
disappear.

Separate the kit into per-instrument stems *first*, and it becomes N easy
questions. Each stem contains one instrument, so onset picking is nearly
trivial, and the stem's own loudness envelope gives a real velocity.

Two independent 2025–26 papers converge on this:

- **[Enhanced ADT via Drum Stem Source Separation](https://arxiv.org/abs/2509.24853)**
  — uses the DrumSep separator to expand ADTOF from **five classes to seven**,
  with **distinct crash and ride**, and derives **per-hit MIDI velocity** from
  per-stem RMS loudness curves.
- **[Separate-and-Detect](https://arxiv.org/html/2608.01093v1)** (posted weeks
  ago) — a five-stem latent-diffusion separator producing kick, snare, toms,
  hi-hats and cymbals, then "a fixed onset detector converts each stem into
  symbolic events". Same shape, newer separator.

The second one also produces *editable stems* as a side effect, which is
directly useful here: the drums-removed track for playing along is the same
artifact.

## What that fixes, specifically

| Complaint | Cause | Fix |
|---|---|---|
| No toms | 5-class model, toms are the weakest class | Toms get their own stem; split floor/mid/high by fundamental pitch |
| No cymbal detection | ADTOF emits its whole cymbal class as GM note 49, literally "crash", so every ride pattern was written as a stream of crashes | The v0.1 separator gives ride and crash their own stems. Without it, spacing and accent now resolve them, so this improves even on the ADTOF path |
| Ghost notes bad | velocity guessed from band energy of a mixed track | Peak loudness in a 50 ms window on the **snare stem alone**, normalised to that stem's own dynamic range |
| Chart displaced by a beat | downbeat phase, not transcription | Cached analysis + an offset control; re-quantizing is instant |

The velocity method is lifted directly from arXiv 2509.24853: peak loudness in a
short window centred on the onset, mapped through a normalised dynamic range.
The important part is *normalised per stem* — a ghost note is quiet relative to
other snare hits, not relative to a kick drum.

## The separator

**DrumSep** (MDX23C / TFC-TDF-Net-v3), by jarredou and aufr33, run through
[ZFTurbo's Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training).
Two checkpoints are published, and the difference matters:

| | Stems | Notes |
|---|---|---|
| **v0.1** (default) | kick, snare, toms, hi-hat, **ride, crash** | SDR 10.8. Separates the cymbals itself. |
| 5-stem | kick, snare, toms, hi-hat, cymbals | Ride vs crash must be guessed downstream. |

Take v0.1. A model trained to tell ride from crash beats inferring it from
spacing and accent, and the six-stem output means that heuristic never runs.

```sh
git clone https://github.com/ZFTurbo/Music-Source-Separation-Training \
  ~/.cache/dcdc/msst
dcdc install-drumsep          # --model 1 for the 5-stem version
```

`dcdc doctor` reports whether it found both the code and a checkpoint.

Canonical URLs live in ZFTurbo's
[pretrained_models.md](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md);
if a release moves, that is the file to check.

## Why not the newest models

- **Noise-to-Notes** (arXiv 2509.21739) is the state of the art on benchmarks
  and has an *inpainting* mode that would be ideal for re-transcribing a few
  bars in place. **No public code or weights.**
- **Separate-and-Detect** (arXiv 2608.01093) is weeks old. No release yet.
- **E-GMD** (444 hours, 43 kits) is the first human-performed dataset with
  velocity annotations, and Onsets-and-Frames adapted to it is a credible
  alternative path. Worth revisiting if DrumSep proves weak.

## Degradation order

The pipeline picks the best installed path and **records which one ran on the
chart**, because a chart missing all its toms because it silently fell back
looks identical to a chart from a bad model — and those need opposite fixes.

1. `stems` — DrumSep + per-stem onsets. Toms, ride vs crash, real velocities.
2. `adtof` — one model, five classes, no velocities.
3. `spectral` — band-energy heuristic. Kick, snare, hi-hat. Nothing else.

If a chart has no toms, check `source.adt_backend` in `chart.json` before
blaming the model.

## Alignment

Alignment is not a transcription problem and should never have been fixed by
re-transcribing. Separation and detection are deterministic for a given
recording; quantization is full of judgement calls.

So the expensive half is cached to `analysis.json`, and `POST
/api/songs/<slug>/requantize` rebuilds the chart with a different downbeat
phase (`offset_beats`), timing nudge (`offset_ms`), or resolution — in
milliseconds, no models.

A "missing half a beat at the start" is `offset_beats`, not a model failure.
