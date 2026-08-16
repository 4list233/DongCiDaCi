"""Command line entry point.

Useful for running the pipeline without the browser -- checking a chart on the
Mac Studio over SSH, batch-transcribing, or debugging a bad transcription
without a web server in the way.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .chart import Chart
from .store import Store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dcdc", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_tr = sub.add_parser("transcribe", help="audio file -> chart.json")
    p_tr.add_argument("audio", type=Path)
    p_tr.add_argument("-t", "--title", default=None)
    p_tr.add_argument("-a", "--artist", default="")
    p_tr.add_argument("-o", "--out", type=Path, default=None, help="chart.json path")
    p_tr.add_argument("--res", type=int, default=None, help="force subdivisions per bar")
    p_tr.add_argument("--adt", default="auto", choices=["auto", "adtof", "spectral"])
    p_tr.add_argument("--grid", default="auto", choices=["auto", "beat_this", "librosa"])

    p_show = sub.add_parser("show", help="print a chart as drum tab")
    p_show.add_argument("chart", type=Path)
    p_show.add_argument("--bars", default="", help="range like 5-12")

    p_serve = sub.add_parser("serve", help="run the web app")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    sub.add_parser("doctor", help="report which pipeline stages are installed")

    args = parser.parse_args(argv)

    if args.command == "transcribe":
        return _transcribe(args)
    if args.command == "show":
        return _show(args)
    if args.command == "serve":
        return _serve(args)
    if args.command == "doctor":
        return _doctor()
    return 1


def _transcribe(args) -> int:
    from . import pipeline

    if not args.audio.exists():
        print(f"no such file: {args.audio}", file=sys.stderr)
        return 1

    title = args.title or args.audio.stem
    work = args.audio.parent / f".{args.audio.stem}-work"

    def progress(stage: str, frac: float) -> None:
        print(f"  [{frac:>4.0%}] {stage}", file=sys.stderr)

    chart, report = pipeline.run(
        audio_path=args.audio, work_dir=work, title=title, artist=args.artist,
        adt_backend=args.adt, grid_backend=args.grid, res=args.res, progress=progress,
    )

    out = args.out or args.audio.with_name("chart.json")
    chart.save(out)

    print(f"\n{len(chart.bars)} bars at res {report.res} -> {out}", file=sys.stderr)
    print(f"grid fit {report.fit_error:.1%}  swing {report.swing:.0%}  "
          f"flams {report.flams}  collisions {report.dropped}", file=sys.stderr)
    for w in report.warnings:
        print(f"  ! {w}", file=sys.stderr)
    return 0


def _show(args) -> int:
    chart = Chart.load(args.chart)
    bars = chart.bars

    if args.bars:
        lo, _, hi = args.bars.partition("-")
        lo_n, hi_n = int(lo), int(hi or lo)
        bars = [b for b in bars if lo_n <= b.n <= hi_n]

    print(f"{chart.title} — {chart.artist or 'unknown'}")
    print(f"{chart.tempo:g} bpm, {chart.time_signature[0]}/{chart.time_signature[1]}, "
          f"res {chart.res}\n")

    from .chart import LANES
    for bar in bars:
        header = f"bar {bar.n}"
        if bar.section:
            header += f"  [{bar.section}]"
        print(header)
        for lane in LANES:
            pattern = bar.lanes.get(lane.key)
            if pattern and set(pattern) != {"-"}:
                print(f"  {lane.key} |{pattern}|")
        if bar.note:
            print(f"   ~ {bar.note}")
        print()
    return 0


def _serve(args) -> int:
    import uvicorn
    uvicorn.run("dcdc.main:app", host=args.host, port=args.port, reload=False)
    return 0


def _doctor() -> int:
    from .pipeline import separate, transcribe

    def mark(ok: bool) -> str:
        return "ok     " if ok else "MISSING"

    try:
        import librosa  # noqa: F401
        has_librosa = True
    except ImportError:
        has_librosa = False
    try:
        import beat_this  # noqa: F401
        has_beat_this = True
    except ImportError:
        has_beat_this = False

    print(f"device            {separate.pick_device()}")
    print(f"demucs            {mark(separate.is_available())}   stage 1, drum isolation")
    print(f"beat_this         {mark(has_beat_this)}   stage 2, preferred beat tracker")
    print(f"librosa           {mark(has_librosa)}   stage 2/3 fallback")
    print(f"adtof             {mark(transcribe.is_adtof_available())}   stage 3, real ADT model")

    if not has_librosa:
        print("\ninstall the pipeline:  pip install -e '.[audio]'")
    elif not transcribe.is_adtof_available():
        print("\nrunning on the spectral fallback -- kick/snare/hi-hat only.")
        print("see docs/SETUP-MAC-STUDIO.md to add ADTOF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
