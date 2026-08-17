"""Command line entry point.

Useful for running the pipeline without the browser -- checking a chart on the
Mac Studio over SSH, batch-transcribing, or debugging a bad transcription
without a web server in the way.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
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
    p_tr.add_argument("--sensitivity", type=float, default=0.5,
                      help="0 keeps only clear hits, 1 keeps nearly everything detected")
    p_tr.add_argument("--adt", default="auto", choices=["auto", "adtof", "spectral"])
    p_tr.add_argument("--grid", default="auto", choices=["auto", "beat_this", "librosa"])

    p_show = sub.add_parser("show", help="print a chart as drum tab")
    p_show.add_argument("chart", type=Path)
    p_show.add_argument("--bars", default="", help="range like 5-12")

    p_serve = sub.add_parser("serve", help="run the web app")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--no-build", action="store_true",
                         help="serve the existing build even if it is out of date")

    sub.add_parser("ls", help="list stored songs")

    p_rm = sub.add_parser("rm", help="delete songs")
    p_rm.add_argument("slugs", nargs="*", help="slugs to delete")
    p_rm.add_argument("--failed", action="store_true", help="delete every failed song")
    p_rm.add_argument("--all", action="store_true", help="delete everything")

    sub.add_parser("doctor", help="report which pipeline stages are installed")

    p_ds = sub.add_parser("install-drumsep", help="download the DrumSep checkpoint")
    p_ds.add_argument(
        "--model", type=int, default=0,
        help="0 = 6 stems with ride and crash separated (default), 1 = 5 stems",
    )

    args = parser.parse_args(argv)

    if args.command == "transcribe":
        return _transcribe(args)
    if args.command == "show":
        return _show(args)
    if args.command == "serve":
        return _serve(args)
    if args.command == "ls":
        return _ls()
    if args.command == "rm":
        return _rm(args.slugs, failed=args.failed, everything=args.all)
    if args.command == "doctor":
        return _doctor()
    if args.command == "install-drumsep":
        return _install_drumsep(args.model)
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

    chart, report, _analysis = pipeline.run(
        audio_path=args.audio, work_dir=work, title=title, artist=args.artist,
        adt_backend=args.adt, grid_backend=args.grid, res=args.res,
        sensitivity=args.sensitivity, progress=progress,
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

    if not args.no_build:
        _ensure_frontend_built()

    uvicorn.run("dcdc.main:app", host=args.host, port=args.port, reload=False)
    return 0


def _ensure_frontend_built() -> None:
    """Rebuild the frontend when the sources are newer than the build.

    The server serves frontend/dist, which is gitignored -- so `git pull` brings
    new interface code and `dcdc serve` keeps serving the build from whenever
    npm was last run. The result is pulling a change and seeing nothing happen,
    which is indistinguishable from the change not working.
    """
    root = Path(__file__).resolve().parents[2] / "frontend"
    dist = root / "dist"
    if not root.exists():
        return

    newest_source = 0.0
    for pattern in ("src/**/*", "index.html", "package.json", "vite.config.*"):
        for path in root.glob(pattern):
            if path.is_file():
                newest_source = max(newest_source, path.stat().st_mtime)

    built = dist / "index.html"
    if built.exists() and built.stat().st_mtime >= newest_source:
        return

    reason = "no build found" if not built.exists() else "the interface has changed"
    npm = shutil.which("npm")
    if not npm:
        print(f"{reason}, and npm is not installed -- the page you get will be "
              f"whatever was last built.\n  build it elsewhere, or pass --no-build "
              f"to silence this.", file=sys.stderr)
        return

    print(f"{reason}; rebuilding the interface...", file=sys.stderr)
    result = subprocess.run([npm, "run", "build"], cwd=root,
                            capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()[-8:]
        print("the rebuild failed, so the page will be the previous build:",
              file=sys.stderr)
        print("\n".join(f"  {line}" for line in tail), file=sys.stderr)
        return
    print("interface rebuilt.", file=sys.stderr)


def _store():
    from .store import Store, default_root
    return Store(default_root())


def _ls() -> int:
    songs = _store().list()
    if not songs:
        print("no songs stored")
        return 0

    width = max(len(s.slug) for s in songs)
    for song in songs:
        size = _song_size(song.slug)
        print(f"{song.slug:<{width}}  {song.status:<8} {size:>7}  {song.title}")
    print(f"\n{len(songs)} song(s), {_human(sum(_bytes(s.slug) for s in songs))} on disk")
    return 0


def _rm(slugs: list[str], failed: bool = False, everything: bool = False) -> int:
    store = _store()
    songs = store.list()

    if everything:
        targets = [s.slug for s in songs]
    elif failed:
        targets = [s.slug for s in songs if s.status == "failed"]
    else:
        targets = list(slugs)

    if not targets:
        print("nothing to delete" if (failed or everything) else "give a slug, --failed, or --all")
        return 0 if (failed or everything) else 2

    known = {s.slug for s in songs}
    missing = [t for t in targets if t not in known]
    if missing:
        print(f"no such song: {', '.join(missing)}")
        return 1

    freed = sum(_bytes(t) for t in targets)
    for slug in targets:
        store.delete(slug)
        print(f"deleted {slug}")
    print(f"\nfreed {_human(freed)}")
    return 0


def _bytes(slug: str) -> int:
    """Size on disk. Separated stems dominate this, not the charts."""
    directory = _store().dir(slug)
    if not directory.exists():
        return 0
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def _song_size(slug: str) -> str:
    return _human(_bytes(slug))


def _human(size: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if size < 1024 or unit == "G":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}G"


def _install_drumsep(model: int) -> int:
    from .pipeline import drumsep

    if model not in range(len(drumsep.CHECKPOINTS)):
        print(f"no such model {model}; choices are:")
        for i, choice in enumerate(drumsep.CHECKPOINTS):
            print(f"  {i}  {choice['name']}  -> {choice['stems']}")
        return 2

    if not (drumsep.MSST_DIR / "inference.py").exists():
        print("the inference code is missing; clone it first:")
        print("  git clone https://github.com/ZFTurbo/Music-Source-Separation-Training "
              f"{drumsep.MSST_DIR}")
        return 1

    if not drumsep.download(model):
        return 1
    print("\ndrumsep installed. re-transcribe a song to use it.")
    return 0


def _doctor() -> int:
    from .pipeline import drumsep, fetch, separate, transcribe

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

    # Having the files is not the same as being able to run them, so this
    # reports runnability rather than mere presence.
    drumsep_ready, drumsep_problems = drumsep.check()

    print(f"device            {separate.pick_device()}")
    print(f"yt-dlp            {mark(fetch.is_available())}   stage 0, audio from a link")
    print(f"demucs            {mark(separate.is_available())}   stage 1, drum isolation")
    print(f"drumsep           {mark(drumsep_ready)}   stage 1b, per-instrument stems")
    print(f"beat_this         {mark(has_beat_this)}   stage 2, preferred beat tracker")
    print(f"librosa           {mark(has_librosa)}   stage 2/3 fallback")
    print(f"adtof             {mark(transcribe.is_adtof_available())}   stage 3, real ADT model")

    if not has_librosa:
        print("\ninstall the pipeline:  pip install -e '.[audio]'")
        return 0

    if not fetch.is_available():
        print("\nno yt-dlp -- links are disabled, audio must be uploaded as a file:")
        print("  pip install -e '.[fetch]'")
    if not has_beat_this:
        print("\nno beat tracker -- downbeats are inferred, so bar 1 may be wrong:")
        print("  pip install 'git+https://github.com/CPJKU/beat_this.git'")
    if not transcribe.is_adtof_available():
        print("\nrunning on the spectral fallback -- kick/snare/hi-hat only, no ghost notes:")
        print("  git clone https://github.com/xavriley/ADTOF-pytorch && pip install -e ADTOF-pytorch")
    if drumsep_ready:
        stems = drumsep.describe_stems()
        if stems:
            print(f"\ndrumsep stems: {stems}")
    else:
        problems = drumsep_problems
        print("\ndrumsep splits the kit into per-instrument stems, which is what "
              "gives you\ntoms, ride vs crash, and real velocities. To finish "
              "setting it up:")
        for problem in problems:
            print(f"  - {problem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
