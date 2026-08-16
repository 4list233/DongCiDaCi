# DongCiDaCi

A drum hobby repo: AI-assisted drum transcription, annotated charts, a fill library, and a
running record of what I'm learning.

The goal is not "push button, get sheet music." It's a workflow where a machine produces a
first draft and **I overwrite it by ear** — so the git history becomes the record of what I
heard, what I chose to play, and how my reading improved.

## Core principle: the chart is a text file

Binary score files (`.mscz`, `.gp`) can't be diffed, grepped, or reviewed in a commit.
Everything here is stored as plain-text notation — LilyPond drum mode
(`\drums`, `DrumStaff`, `DrumVoice`) — so charts version like code.

MusicXML is used only to move data *between* tools, never to store work.

## The pipeline

| # | Stage | Tool | What breaks |
|---|-------|------|-------------|
| 1 | Isolate the kit | Demucs v4 (`htdemucs`) | Separation artifacts erase ghost notes |
| 2 | Find the grid | Beat This! / All-In-One | A downbeat off by one shifts the whole chart |
| 3 | Detect + classify | ADTOF (5 classes) | Ride and crash collapse into one class |
| 4 | Quantize to notation | music21 / partitura | No off-the-shelf answer for drums — custom code |
| 5 | Engrave + correct | LilyPond | Nothing; this is the human part |

```sh
# 0 — check Songsterr first; a human transcription to disagree with beats
#     a machine transcription to verify from zero

# 1 — isolate the kit
python -m demucs --two-stems=drums -n htdemucs song.mp3

# 2 — establish the grid (beats, downbeats, section map)
allin1 predict song.mp3

# 3 — transcribe the isolated stem (ADTOF notebook) -> raw.mid

# 4 — quantize MIDI against the stage-2 downbeats -> score.ly

# 5 — engrave, then fix by ear
lilypond songs/<song>/score.ly
```

## Layout

```
songs/<song>/
  source.md    # link, tempo, meter, tuning notes
  raw.mid      # machine output, never edited
  score.ly     # my chart — the source of truth
  score.pdf
  notes.md     # what the AI got wrong, what I chose to play

fills/         # one snippet per fill, tagged by genre + subdivision
gear/          # cymbals and kit, and what each is for
genres/        # per-style notes (trap, etc.)
practice/      # session log, links into songs/
docs/research/ # landscape research
```

Keeping `raw.mid` immutable next to an edited `score.ly` means `git diff` answers
"what did I change, and why" for every song.

## Research

- [Drum transcription landscape (Aug 2026)](docs/research/2026-08-drum-transcription-landscape.md)
  — open-source repos, research frontier, prior art, and what every model still gets wrong.
