# DongCiDaCi

A drum hobby repo, and the web app that feeds it: AI-assisted transcription,
annotated charts, a fill library, and a running record of what I'm learning.

The goal is not "push button, get sheet music." A machine produces a first
draft; **I overwrite it by ear**. The git history is the record of what I heard,
what I chose to play, and how my reading improved.

```
audio ──▶ separate ──▶ grid ──▶ transcribe ──▶ quantize ──▶ chart.json ──▶ browser
                                                                │
                                                       edit ◀───┘  play along
```

## Quickstart

macOS ships Python 3.9 at best, which is too old, so the prerequisites are not
optional:

```sh
brew install python@3.12 node ffmpeg
```

Use **3.12**, not 3.13 — Demucs 4.x predates 3.13 and has no wheels for it.
`ffmpeg` is not optional either; librosa and Demucs both shell out to it for
anything that is not a WAV.

```sh
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[audio]'          # ~2GB, pulls torch
npm --prefix frontend install && npm --prefix frontend run build
dcdc doctor                        # which stages are live
dcdc serve                         # http://127.0.0.1:8000
```

To see the app before committing to a 2GB download, `pip install -e .` installs
the API and editor alone in seconds. The pipeline stages will report as missing,
which is exactly what `dcdc doctor` is for.

Full Apple Silicon instructions, including the two optional model upgrades, are
in [docs/SETUP-MAC-STUDIO.md](docs/SETUP-MAC-STUDIO.md).

## How it works

**A Spotify link cannot give you audio.** The Audio Features and Audio Analysis
endpoints were closed to new apps in November 2024, and the Web Playback SDK is
DRM'd specifically so you cannot reach the samples. Paste the link for metadata;
supply the audio yourself.

Five stages. Stage 5 is the browser, and it is the one that matters.

| # | Stage | Tool | Failure mode |
|---|-------|------|--------------|
| 1 | Isolate the kit | Demucs v4 (`htdemucs`, MPS) | Artifacts erase ghost notes |
| 2 | Find the grid | Beat This! → librosa | A downbeat off by one displaces the whole chart |
| 3 | Detect + classify | ADTOF → spectral fallback | 5 classes; ride and crash collapse into one |
| 4 | Quantize to notation | custom | No off-the-shelf answer — this is the real work |
| 5 | Engrave + correct | alphaTab in the browser | None; this is the part you do |

Stages 2 and 3 degrade rather than fail. If ADTOF is not installed you get a
spectral fallback that finds kick, snare and hi-hat and nothing else — and the
app says so in a banner, because believing you got a real transcription is the
worst available outcome.

## The chart format

`chart.json` is the canonical, hand-editable source of truth. Everything else —
AlphaTex for the browser, MIDI for a DAW — is generated and disposable.

```json
{"n": 5, "section": "verse", "lanes": {
  "hh": "x-x-x-x-x-x-x-x-",
  "sd": "----o---g---o---",
  "bd": "o--o----o-------"}}
```

One character per subdivision, one bar per line. The vocabulary is drum tab,
because drummers already read it and because it makes `git diff` legible — you
can see the groove change. It also makes the fill library greppable:

```sh
grep -l '"sd": "..oo..oo' fills/*.json
```

| char | meaning | | char | meaning |
|------|---------|-|------|---------|
| `-` | rest | | `o` | hit (drums) |
| `x` | hit (cymbals) | | `O` | accent |
| `X` | accent | | `g` | ghost note |
| `+` | open hi-hat | | `f` | flam |

## Layout

```
songs/<song>/
  song.json    job state and metadata
  source.mp3   your audio            (gitignored)
  stems/       demucs output         (gitignored)
  raw.json     machine's first draft (committed, never edited)
  chart.json   your chart            (committed, the truth)
  notes.md     what the AI missed

fills/         one snippet per fill, tagged by genre + subdivision
gear/          cymbals and kit, and what each is for
genres/        per-style notes (trap, etc.)
practice/      session log
docs/research/ landscape research
```

`raw.json` is written once and never overwritten, so `git diff raw.json chart.json`
answers "what did the machine get wrong" for any song.

## Development

```sh
.venv/bin/pytest backend/tests      # 45 tests, no ML deps or GPU required
npm --prefix frontend test          # parses generated AlphaTex through alphaTab
npm --prefix frontend run dev       # Vite on :5173, proxies /api to :8000
```

The AlphaTex test is not ceremonial. alphaTab validates percussion articulation
names against a fixed vocabulary and rejects the **entire score** if one is
wrong — a plausible-but-invented name produces a blank page, not a missing note.

CLI, for working without the browser:

```sh
dcdc doctor                         # which stages are installed
dcdc transcribe song.mp3            # audio -> chart.json
dcdc show chart.json --bars 5-12    # print as drum tab
```

## Research

[Drum transcription landscape (Aug 2026)](docs/research/2026-08-drum-transcription-landscape.md)
— open-source repos assessed, research frontier, prior art, and what every model
still gets wrong.
