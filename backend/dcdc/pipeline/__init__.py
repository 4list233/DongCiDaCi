"""The four machine stages. Stage 5 -- engraving and correction -- is the browser."""

from . import grid, quantize, separate, transcribe  # noqa: F401

__all__ = ["separate", "grid", "transcribe", "quantize", "run"]


def run(audio_path, work_dir, title, artist="", adt_backend="auto",
        grid_backend="auto", res=None, progress=None):
    """Audio in, Chart out. Progress is reported as (stage_name, fraction)."""
    from pathlib import Path

    audio_path = Path(audio_path)
    work_dir = Path(work_dir)

    def step(name, frac):
        if progress:
            progress(name, frac)

    step("separating drums", 0.05)
    sep = separate.separate(audio_path, work_dir / "stems")

    step("finding the grid", 0.55)
    beat_grid = grid.detect(audio_path, backend=grid_backend)

    step("transcribing", 0.70)
    transcription = transcribe.transcribe(sep.drums, backend=adt_backend)

    step("quantizing", 0.90)
    chart, report = quantize.quantize(
        transcription, beat_grid, title=title, artist=artist, res=res
    )
    chart.source["separation_model"] = sep.model
    chart.source["separation_device"] = sep.device

    step("done", 1.0)
    return chart, report
