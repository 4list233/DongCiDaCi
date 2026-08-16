# Drum Transcription Landscape

*Research dive, August 2026 — open-source repositories, research frontier, and prior art.*

Everything currently worth building on if the goal is turning songs into drum charts you can
annotate, argue with, and keep.

---

## 1. The decision everything else hangs off: make the chart a text file

Annotating, versioning, and searching a fill library all rule out a binary score file as the
source of truth. `.mscz` and `.gp` can't be diffed, grepped, or reviewed in a commit.

The two credible plain-text options for drums:

- **LilyPond drum mode** — `\drums`, `DrumStaff`, `DrumVoice`, with per-instrument notehead and
  staff-position control. Publication-grade PDF output, fully scriptable. Source reads almost
  like the pattern itself: `hh8 hh sn hh`.
- **AlphaTex** — alphaTab's plain-text format, renders live in a browser with playback.

**MusicXML** sits in the middle as the universal interchange format — every tool reads and
writes it — but it's verbose XML and miserable to hand-edit. Use it to move between tools,
not to store work.

The shape that follows: audio in → machine produces MIDI → MIDI converted *once* into text
notation → from that moment the text file is the score. The AI output is a first draft you
overwrite, and your corrections live in git history.

---

## 2. Five stages, and everyone builds the same five

Read enough of these repos and the same pipeline reappears. Projects differ almost entirely
in which stage they invest in and which they hand-wave.

| # | Stage | Best open tool | Failure mode |
|---|-------|----------------|--------------|
| 1 | **Isolate the drums** — strip vocals/bass/guitar so the detector isn't fighting the mix | Demucs v4 `htdemucs` | Bleed and separation artifacts smear soft ghost notes into nothing |
| 2 | **Find the grid** — beats, downbeats, tempo, meter | Beat This! / All-In-One | A downbeat off by one puts the whole chart on the "and" |
| 3 | **Detect + classify** — onsets, then which piece of the kit | ADTOF / Omnizart | Most models resolve only 3–5 classes; ride and crash become one thing |
| 4 | **Quantize to notation** — snap hits to subdivisions against the stage-2 grid | music21 / partitura | Over-quantize and feel dies; under-quantize and you get unreadable 128ths |
| 5 | **Engrave + annotate** | LilyPond / alphaTab | None — this is the part you do, and the part you keep |

Stage 1 has research backing: *Enhanced Automatic Drum Transcription via Drum Stem Source
Separation* (2025) is essentially the formal argument that running separation before
transcription improves accuracy. Every serious hobbyist workflow found in the wild already
does this by instinct.

**Stage 4 has no off-the-shelf answer for drums.** Both music21 and partitura explicitly warn
that a MIDI *performance* won't load cleanly as a *score*. That gap is the real project;
everything else is assembly.

---

## 3. Open-source inventory

Rated on whether you could get useful output this weekend, not on research merit. Two of the
most-cited projects in this space are effectively archival.

### Works today

**[Demucs](https://github.com/facebookresearch/demucs)** — Meta's music source separation.
v4's hybrid transformer (`htdemucs`) hits 9.20 dB SDR on MUSDB HQ; `htdemucs_ft` is ~4× slower
and slightly better. *The single highest-leverage tool in the chain and the least likely to
break.*

**[Beat This!](https://github.com/CPJKU/beat_this)** — Transformer beat/downbeat tracker,
ISMIR 2024, notable for dropping the traditional DBN post-processing. C++ port exists.
*The cleanest replacement for madmom's aging beat tracker.*

**[All-In-One Music Structure Analyzer](https://github.com/mir-aidj/all-in-one)** — Beats,
downbeats, tempo *and* section boundaries with labels (intro/verse/chorus) in one pass; can
sonify its analysis over the original audio. *Underrated for charting: section labels give you
rehearsal marks for free, and fills cluster on section transitions.*

**[alphaTab](https://github.com/CoderLine/alphaTab)** — Cross-platform notation/tab rendering.
Reads Guitar Pro 3–7, MusicXML, and AlphaTex; renders SVG in-browser; plays back via alphaSynth.
Handles drum tracks in both tab and standard notation. *The viewer, if you go web.*

**[LilyPond](https://lilypond.org/doc/v2.24/Documentation/notation/common-notation-for-percussion)**
— GNU engraver with first-class percussion support. Imports MusicXML. *The engraver.*

**[music21](https://music21.org/) / [partitura](https://github.com/CPJKU/partitura)** — music21
(MIT) is mature and elaborate, exports MusicXML and LilyPond; partitura (CPJKU) is lighter with
`load_score_midi()` offering built-in quantization. *Stage-4 glue.*

**[Groove MIDI Dataset](https://magenta.withgoogle.com/datasets/groove) +
[GrooVAE](https://magenta.tensorflow.org/groovae)** — 13.6 hours of professionally played,
tempo-aligned drumming: 1,150 MIDI files, 22,000+ measures, 10 drummers on a Roland TD-11,
deliberately mixing long grooves with **short beats and fills**. GrooVAE adds humanize,
tap2drum, and interpolation. *This is the fill-inspiration engine — a searchable corpus of real
fills with velocity and microtiming intact, far more useful than generating fills from scratch.*

### Usable, fiddly

**[ADTOF](https://github.com/MZehren/ADTOF)** — The strongest openly available ADT model.
Trained on 359 hours of real (non-synthetic) music crowdsourced from rhythm-game charts.
5 classes: kick, snare, hi-hat, toms, cymbals. *The transcription engine.*
Caveats: Python 3.10, TensorFlow/Keras/madmom, driven from a notebook rather than a clean CLI,
and licensed **CC BY-NC-SA** (fine for a hobby, not for anything commercial). An
`ADTOF-pytorch` variant drops the TF/madmom dependency for ~0.2% F-measure. Dataset on Zenodo
(`10.5281/zenodo.10084510`).

**[transcribir-bateria](https://github.com/gaston-michel/transcribir-bateria)** — The most
complete end-to-end open example: Streamlit → Demucs → librosa onsets → ResNet CNN
(kick/snare/hat/crash) → **Abjad → LilyPond → PDF**, exporting PDF, MIDI, and LilyPond source.
MIT. A 2024 university project with 3 stars, so don't run it as-is — but it independently
arrived at the Abjad/LilyPond text-source-of-truth design. *Best architectural reference here.*

**[DrummerScore](https://github.com/skittree/DrummerScore)** — Demucs + Torch/librosa ADT with
a FastAPI + HTMX front end and a spectrogram labelling UI (pigeonXT). MP3 in, labelled MIDI out.
MIT, 31 stars. Stops at MIDI rather than notation, but *its labelling interface is the closest
anyone has come to the human-in-the-loop correction step.*

**[DrumSheet](https://github.com/MLecardonnel/DrumSheet)** — ResNet50V2 per-instrument binary
classifiers plus a custom cymbal CNN, trained on annotated MedleyDB (7,994 onsets, 23 tracks).
LilyPond → PDF. Apache-2.0. Reports **87.8% average detection with 1.95% false positives** —
a realistic yardstick, and a reminder that ~12% of hits need your ear.

**[AI-Drum-Transcriber-Workflow](https://github.com/Gary-nope/AI-Drum-Transcriber-Workflow)** —
Not a library, a documented workflow. Path 1: skip AI entirely, pull an existing Songsterr tab.
Path 2: Demucs (Colab/UVR5/X-Minus) → DAW drum triggers → GM channel-10 MIDI. Zero stars and
incomplete docs, but *Path 1's "check for a human transcription first" instinct is correct.*

### Archival — read, don't depend on

**[Omnizart](https://github.com/Music-and-Culture-Technology-Lab/omnizart)** — Ambitious
all-instrument toolkit (drums, vocal, chords, beat), JOSS-published, CNN drum model ~9.4M params.
Flagged inactive; the drum model has acknowledged bugs preventing training convergence
(checkpoints work); installation is painful on Apple Silicon and Windows because of madmom.
Docker image exists if you must.

**[ADTLib](https://github.com/CarlSouthall/ADTLib)** — The classic. 209 stars, BSD-2,
`ADT Drum.wav` on the command line, outputs onset text plus auto-generated tab PDF via fpdf.
Kick/snare/hi-hat only, TensorFlow 1.x era, unpinned deps. *Its interface is still the best in
the field — one command, tab out. Worth stealing that ergonomic idea.*

---

## 4. Research frontier (2026)

**Noise-to-Notes (N2N)** ([arXiv 2509.21739](https://arxiv.org/abs/2509.21739)) is the current
state of the art. It reframes drum transcription as conditional generation: audio-conditioned
Gaussian noise diffused into drum onsets *with velocities*. Uses an annealed pseudo-Huber loss
to jointly optimise binary onsets and continuous velocity, plus features from music foundation
models for robustness on out-of-domain audio. Sets new benchmarks, offers a speed/accuracy dial
and **inpainting**.

Inpainting is the interesting part: re-transcribe just bars 33–40 while holding the rest fixed.
That's a *correction* primitive, not just a transcription one — exactly the shape a
human-in-the-loop workflow wants.

**No public code or weights surfaced.** Treat N2N as a preview of where hobbyist tooling lands
in a year or two, not something to build on now.

Two other threads:

- **Synthetic / semi-supervised training** — recent work reports state-of-the-art on the ENST
  and MDB test sets using synthetic data, beating fully supervised methods. The labelled-data
  bottleneck is easing.
- **Better datasets** — *STAR Drums* (TISMIR) joins ADTOF as a modern benchmark. Dataset
  quality, not architecture, has been the limiting factor in this field for a decade.

*Practical read:* installable open tooling lags the papers by roughly two to three years.
ADTOF is the most recent thing that crossed the gap into usable software.

---

## 5. What every model still gets wrong

This defines what the annotation layer is *for*. These aren't bugs to wait out — they're
structural, and they're precisely the musical decisions worth recording.

| Failure | Why | What to capture in `notes.md` / the chart |
|---------|-----|------------------------------------------|
| **Ghost notes** | Low-velocity snare buried under separation artifacts; models optimised for F-measure learn to drop them | Ghosts *are* the groove — mark them explicitly |
| **Hi-hat articulation** | Open/closed/half-open/foot/splash are near-identical spectrally after separation; most models emit one "hi-hat" class | Open/closed markings, pedal notation below the staff |
| **Ride vs crash vs china** | ADTOF's 5 classes collapse all cymbals into one; 3-class models have no cymbal class beyond the hat | Which cymbal, where on it (bow/edge/bell) |
| **Sticking / limb assignment** | Audio simply doesn't contain it — identical audio, different hands | R/L sticking, which tom is which in a fill. Pure human authorship |
| **Rudiments** | Flams, drags, buzz rolls arrive as onset clusters milliseconds apart, then quantize into nonsense | Collapse the cluster into the rudiment it actually is |
| **Dynamics and feel** | Velocity is usually discarded en route to notation; swing and pushed/laid-back timing get flattened | Accents, dynamics, a note on where the pocket sits |
| **Tempo drift** | Human-played records aren't click-locked; a fixed grid desynchronises across a song | Nothing — fix upstream with a proper beat tracker |

A drummer's published comparison of AI transcription services
([Francis' Drumming Blog](https://francisdrummingblog.com/2024/01/23/ai-generative-drum-transcriptions-a-comparative-analysis/),
updated 2026) lands in the same place, as does Klangio's own documentation: Drum2Notes is their
strongest product, "often near-perfect on isolated drum recordings", and still ships with the
caveat that results depend heavily on mix and recording quality.

---

## 6. Prior art outside GitHub

**[Soundslice](https://www.soundslice.com/practice-drums/)** — closest thing to the goal.
Notation editor with drum-kit tracks in tab *or* standard notation, synced to YouTube or MP3 via
a syncpoint editor (tap downbeats while it plays). Click a note to jump there, drag to loop,
slow down, solo parts, toggle sticking. *If nothing else gets built, build the syncpoint idea:
a chart that knows where it is in the recording.*

**[Klangio Drum2Notes](https://klang.io/drum2notes/)** — dedicated AI drum transcriber; exports
PDF, MusicXML, and both quantized and unquantized MIDI. *Use as the accuracy yardstick. The
unquantized export is the honest one, and MusicXML out means its results can feed this pipeline.*

**[Moises](https://moises-ai.com/)** — stem separation plus practice tooling (loops, isolation,
speed). Endorsed by Jay Weinberg and Drumeo instructors; Eloy Casagrande used it to prepare 32
songs for the Slipknot audition. *Validates the separate-first habit.*

**[Songsterr](https://www.songsterr.com/)** — large library of human-made tabs including drums,
with playback and per-track isolation. *Cheapest possible first step: if a human already charted
the song, spend effort disagreeing with their reading rather than re-deriving it.*

Also worth knowing: **MuseScore** MIDI import has adaptive quantization (grid resolution
selectable down to 128ths, with a "reduce rests" option for drum tracks) — but the MIDI import
panel from MuseScore 3 was **not** carried into MuseScore 4, so quantization control there is
weaker than it used to be.

---

## 7. Notation conventions

Standard drum notation vs. tab is a real fork, not a style preference. Tab uses proportional
horizontal placement and struggles to convey dynamics, ghost notes, and note durations.
Standard notation carries all of it. For a workflow whose whole point is capturing *ghost notes,
articulation, and dynamics*, standard notation on a 5-line percussion staff is the only option
that can hold the information.

Ted Reed's *Syncopation* and similar method books are the reference for how readable drum
notation is laid out — worth matching those conventions so charts read like the books already
on the shelf.

---

## 8. Open questions

- Stage 4 (MIDI → readable notation) needs custom code. Does the grid come from Beat This! or
  All-In-One, and how are swung 8ths detected rather than quantized away?
- Is a correction UI (à la DrummerScore's labelling interface) worth building, or is editing
  LilyPond source directly faster once fluent?
- Where does the fill library live — LilyPond snippets, MIDI, or both — and what's the tagging
  scheme (genre / subdivision / bar length / limb pattern)?
