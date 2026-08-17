"""The machine stages. Stage 5 -- correction and playing parts in -- is the browser."""

from . import analysis, drumsep, fetch, grid, onsets, quantize, separate, transcribe  # noqa: F401

__all__ = [
    "fetch", "separate", "drumsep", "grid", "onsets", "transcribe",
    "quantize", "analysis", "run", "requantize",
]


def run(audio_path, work_dir, title, artist="", adt_backend="auto",
        grid_backend="auto", res=None, progress=None):
    """Audio in, (Chart, report, Analysis) out.

    The transcription path is chosen by what is installed, in descending order
    of quality:

      1. per-stem  DrumSep splits the kit, then each stem is onset-picked.
                   Gives toms, ride vs crash, and real velocities.
      2. ADTOF     one model, five classes, no velocities.
      3. spectral  band-energy heuristic. Kick, snare, hi-hat only.

    Which one actually ran is recorded on the chart, because a chart missing all
    its toms because it silently fell back looks exactly like a chart from a bad
    model, and those need completely different fixes.
    """
    from pathlib import Path

    audio_path = Path(audio_path)
    work_dir = Path(work_dir)

    def step(name, frac):
        if progress:
            progress(name, frac)

    step("separating drums", 0.05)
    sep = separate.separate(audio_path, work_dir / "stems")

    step("splitting the kit", 0.35)
    stems = drumsep.separate(sep.drums, work_dir / "stems")

    step("finding the grid", 0.55)
    beat_grid = grid.detect(audio_path, backend=grid_backend)

    step("transcribing", 0.70)
    if stems and adt_backend in ("auto", "stems"):
        found, stem_warnings = onsets.transcribe_stems(
            stems.lane_stems(), split_cymbal_stem=stems.cymbals_need_splitting
        )
        notes = [f"per-stem transcription via {stems.model}"]
        if stems.cymbals_need_splitting:
            notes.append(
                "this model emits one cymbal stem, so ride vs crash is inferred "
                "from spacing and accent -- check the fills"
            )
        transcription = transcribe.Transcription(
            onsets=found, backend="stems", warnings=stem_warnings + notes,
        )
    else:
        transcription = transcribe.transcribe(sep.drums, backend=adt_backend)
        if adt_backend == "auto":
            # Say which of the two very different situations this is. A chart
            # with no toms because separation was never installed and one
            # because separation was installed and failed look identical, and
            # they need opposite fixes.
            if drumsep.is_available():
                # Quote the real failure. Pointing at `dcdc doctor` is useless
                # when doctor is clean and the break only happens at run time.
                transcription.warnings.append(
                    "DrumSep is installed but produced no stems, so this fell back "
                    "to whole-kit transcription: "
                    + (stems.error or "no reason reported")
                )
            else:
                transcription.warnings.append(
                    "DrumSep not installed -- no per-instrument stems, so toms and "
                    "cymbals are approximated and velocities are estimates"
                )

    step("quantizing", 0.90)
    chart, report = quantize.quantize(
        transcription, beat_grid, title=title, artist=artist, res=res
    )
    chart.source.update({
        "separation_model": sep.model,
        "separation_device": sep.device,
        "kit_model": stems.model or "",
        "adt_backend": transcription.backend,
    })

    record = analysis.Analysis(
        onsets=transcription.onsets,
        grid=beat_grid,
        adt_backend=transcription.backend,
        grid_backend=beat_grid.backend,
        warnings=report.warnings,
    )

    step("done", 1.0)
    return chart, report, record


def requantize(record, title, artist="", res=None, offset_beats=0, offset_ms=0.0):
    """Rebuild a chart from cached analysis. Milliseconds, no models."""
    shifted = record.shifted(offset_beats=offset_beats, offset_ms=offset_ms)
    chart, report = quantize.quantize(
        shifted.transcription(), shifted.grid, title=title, artist=artist, res=res
    )
    chart.source.update({
        "adt_backend": record.adt_backend,
        "offset_beats": offset_beats,
        "offset_ms": offset_ms,
    })
    return chart, report
