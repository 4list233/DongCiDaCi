"""Turning detected sound into a part a person could play.

Detection answers "what made a noise". A chart has to answer something harder:
what was the drummer *doing*. Those differ, and the gap is where a transcription
stops being useful.

Three things separate them, and all three are physical rather than acoustic:

  bleed      separated stems are not clean. The toms stem carries kick and snare
             leakage, the cymbal stems carry hat. Detected at a low threshold,
             that leakage becomes notes -- an accurate description of the audio
             and a wrong description of the performance.

  limbs      a drummer has two hands and two feet. Five simultaneous voices is
             not a hard part, it is an impossible one, so it is evidence of
             over-detection rather than of virtuosity.

  intent     a hit so much quieter than everything around it that no listener
             would notice it is not part of the part. Keeping it is faithful to
             the waveform and unfaithful to the music.

None of this is guesswork about style. It is the constraint that the thing being
transcribed was played by a body.
"""

from __future__ import annotations

import logging

from .transcribe import Onset

log = logging.getLogger(__name__)

# Hits closer together than this were struck at the same instant. Around 25ms is
# where a listener stops hearing two events and hears one.
COINCIDENCE_S = 0.025

# Within one instant, a voice this much quieter than the loudest is leakage from
# it rather than a hit of its own. A real simultaneous hit -- kick with crash --
# is played with intent, and both land hard.
BLEED_RATIO = 0.4

# Hands available. The feet are counted separately because they have their own
# dedicated lanes.
MAX_HANDS = 2

FOOT_LANES = frozenset({"bd", "hf"})

# Which lane a hand keeps when it has to choose. A backbeat matters more than the
# hat pattern under it, and a crash marks a section where a tom is just colour.
HAND_PRIORITY: dict[str, int] = {
    "sd": 5,      # the backbeat is the part
    "cc": 4,      # accents mark structure
    "rd": 3,
    "hh": 3,
    "ht": 2, "mt": 2, "lt": 2,
    "ho": 3,
}


def clean(
    onsets: list[Onset],
    sensitivity: float = 0.5,
    max_hands: int = MAX_HANDS,
) -> tuple[list[Onset], dict[str, int]]:
    """Reduce detected onsets to a playable part.

    `sensitivity` runs 0 to 1. Low keeps only what is clearly played; high keeps
    nearly everything detected. It is applied here rather than during detection
    so it can be retuned instantly from cached analysis, without re-running a
    minute of separation to find out whether 0.4 reads better than 0.6.

    Returns the surviving onsets and a count of what each stage removed, because
    "we deleted 80 of your notes" is something a chart should admit to.
    """
    removed = {"quiet": 0, "bleed": 0, "limbs": 0}
    if not onsets:
        return [], removed

    kept = _drop_quiet(sorted(onsets, key=lambda o: o.time), sensitivity, removed)
    kept = _drop_bleed(kept, removed)
    kept = _limit_limbs(kept, max_hands, removed)
    return kept, removed


def _drop_quiet(onsets: list[Onset], sensitivity: float, removed: dict) -> list[Onset]:
    """Remove hits too quiet to be part of the part.

    The floor is per lane, because loudness only means something relative to the
    same drum: a ghost note is quiet compared to other snare hits, not compared
    to a kick. This is the same reasoning that makes velocity worth measuring per
    stem in the first place.
    """
    sensitivity = min(max(sensitivity, 0.0), 1.0)
    # At 1.0 nothing is dropped; at 0.0 only hits near the lane's own loudest
    # survive. 0.5 lands around a fifth of peak, which keeps real ghost notes.
    floor_fraction = (1.0 - sensitivity) * 0.55

    by_lane: dict[str, list[Onset]] = {}
    for onset in onsets:
        by_lane.setdefault(onset.lane, []).append(onset)

    survivors = []
    for lane, group in by_lane.items():
        peak = max(o.velocity for o in group) or 1.0
        floor = peak * floor_fraction
        for onset in group:
            if onset.velocity >= floor:
                survivors.append(onset)
            else:
                removed["quiet"] += 1

    survivors.sort(key=lambda o: o.time)
    return survivors


def _drop_bleed(onsets: list[Onset], removed: dict) -> list[Onset]:
    """Within one instant, drop voices that are just leakage from a louder one."""
    survivors = []
    for cluster in _clusters(onsets):
        if len(cluster) == 1:
            survivors.extend(cluster)
            continue

        loudest = max(o.velocity for o in cluster)
        threshold = loudest * BLEED_RATIO
        for onset in cluster:
            # Never let leakage reasoning delete the loudest voice of an instant.
            if onset.velocity >= threshold:
                survivors.append(onset)
            else:
                removed["bleed"] += 1
    return survivors


def _limit_limbs(onsets: list[Onset], max_hands: int, removed: dict) -> list[Onset]:
    """Keep each instant playable by two hands and two feet."""
    survivors = []
    for cluster in _clusters(onsets):
        feet = [o for o in cluster if o.lane in FOOT_LANES]
        hands = [o for o in cluster if o.lane not in FOOT_LANES]

        # Both feet can play at once, but neither can play two things at once.
        kept_feet: dict[str, Onset] = {}
        for onset in feet:
            best = kept_feet.get(onset.lane)
            if best is None or onset.velocity > best.velocity:
                if best is not None:
                    removed["limbs"] += 1
                kept_feet[onset.lane] = onset
            else:
                removed["limbs"] += 1

        if len(hands) > max_hands:
            # Musical importance first, loudness to break ties.
            hands.sort(
                key=lambda o: (HAND_PRIORITY.get(o.lane, 1), o.velocity),
                reverse=True,
            )
            removed["limbs"] += len(hands) - max_hands
            hands = hands[:max_hands]

        survivors.extend(list(kept_feet.values()) + hands)

    survivors.sort(key=lambda o: o.time)
    return survivors


def _clusters(onsets: list[Onset]) -> list[list[Onset]]:
    """Group onsets struck at effectively the same moment."""
    clusters: list[list[Onset]] = []
    current: list[Onset] = []
    for onset in onsets:
        if current and onset.time - current[0].time > COINCIDENCE_S:
            clusters.append(current)
            current = []
        current.append(onset)
    if current:
        clusters.append(current)
    return clusters


def describe(removed: dict[str, int], kept: int) -> list[str]:
    """What was thrown away, in terms worth reading."""
    notes = []
    if removed.get("bleed"):
        notes.append(
            f"dropped {removed['bleed']} hit(s) that looked like bleed between stems"
        )
    if removed.get("limbs"):
        notes.append(
            f"dropped {removed['limbs']} hit(s) that would have needed a third hand"
        )
    if removed.get("quiet"):
        notes.append(
            f"dropped {removed['quiet']} hit(s) below the sensitivity floor -- "
            "raise sensitivity if the part is missing detail"
        )
    total = sum(removed.values())
    if total > kept:
        notes.append(
            f"more hits were removed ({total}) than kept ({kept}) -- detection is "
            "probably picking up resonance, so try a lower sensitivity"
        )
    return notes
