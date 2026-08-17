/**
 * Chart -> AlphaTex.
 *
 * AlphaTex is alphaTab's plain-text notation format. It is a *render target*
 * here, not a storage format: chart.json is the source of truth and this is
 * regenerated on every edit. That means we never have to parse AlphaTex back,
 * which matters because alphaTab's own docs warn that reaching into its data
 * model tends to break the rendering pipeline.
 */

// Lane key -> alphaTab percussion articulation. Mirrors backend/dcdc/chart.py;
// the editor fetches the authoritative list from /api/vocabulary at boot and
// overwrites this, so a mismatch shows up as a render bug, not silent drift.
//
// These names come from alphaTab's fixed percussion vocabulary. Anything not in
// it fails the whole score at parse time, so they are verified by
// frontend/test/alphatex.test.mjs rather than trusted.
export const DEFAULT_ARTICULATIONS = {
  cc: 'CrashHighHit',
  rd: 'RideMiddle',
  hh: 'HiHatClosed',
  ht: 'HighTomHit',
  mt: 'MidTomHit',
  lt: 'LowFloorTomHit',
  sd: 'SnareHit',
  bd: 'KickHit',
  hf: 'PedalHiHatHit',
};

// Characters that mean "the hi-hat is open" rather than closed.
const OPEN_CHARS = new Set(['+']);
const REST = '-';

/**
 * Note value for one slot, plus a tuplet marker when the grid is triplet-based.
 *
 * Straight grids map directly: 4 slots per beat in 4/4 is a 16th note. Triplet
 * grids do not -- 3 slots per beat is not a "12th note", it is three 8th notes
 * carrying a tuplet bracket. Emitting `.12` produces an invalid duration and
 * alphaTab rejects the whole score.
 */
function slotValue(res, timeSignature) {
  const [beats, unit] = timeSignature;
  const perBeat = res / beats;

  if (perBeat % 3 === 0) {
    // Three notes in the space of two: written as the value that fits two per
    // beat at this depth, bracketed as a triplet.
    return { duration: (unit * perBeat * 2) / 3, tuplet: 3 };
  }
  return { duration: perBeat * unit, tuplet: null };
}

/** Backwards-compatible helper for callers that only need the note value. */
function slotDuration(res, timeSignature) {
  return slotValue(res, timeSignature).duration;
}

/**
 * Note values available for a run of N slots, largest first.
 *
 * A run of 2 slots at res 16 is an eighth note, 3 is a dotted eighth, 4 a
 * quarter. Ties do not work on a percussion staff in alphaTab, so anything not
 * directly expressible (5, 7, 9 slots...) takes the largest value that fits and
 * the remainder becomes rests.
 */
function noteValues(slotDen) {
  const values = [];
  for (let k = 0; k <= 4; k++) {
    const denominator = slotDen >> k;
    if (denominator < 1) break;
    values.push({ slots: 1 << k, duration: denominator, dots: '' });
    if (k >= 1) {
      values.push({ slots: 1.5 * (1 << k), duration: denominator, dots: '{d}' });
    }
  }
  return values.sort((a, b) => b.slots - a.slots);
}

/** Greedily cover `slots` with the largest available note values. */
function cover(slots, values) {
  const out = [];
  let left = slots;
  while (left > 0) {
    const fit = values.find((v) => v.slots <= left);
    if (!fit) break;                       // cannot happen: 1 slot always fits
    out.push(fit);
    left -= fit.slots;
  }
  return out;
}

/** The articulation names sounding at one slot, with their effects applied. */
function hitsAt(bar, slot, articulations) {
  const hits = [];
  for (const [lane, pattern] of Object.entries(bar.lanes || {})) {
    const ch = pattern[slot];
    if (!ch || ch === REST) continue;

    let name = articulations[lane];
    if (!name) continue;
    if (lane === 'hh' && OPEN_CHARS.has(ch)) name = 'HiHatOpen';

    // Accents and ghosts are effects on the note, not different notes.
    let effects = '';
    if (ch === 'X' || ch === 'O') effects = '{ac}';
    else if (ch === 'g') effects = '{g}';
    else if (ch === 'f') effects = '{gr}';

    hits.push(`${name}${effects}`);
  }
  return hits;
}

/**
 * Render one bar as AlphaTex beats.
 *
 * Notes are held until the next event rather than written as one note plus a
 * rest per empty slot. That distinction is the whole readability of the chart:
 * a plain eighth-note hi-hat pattern written on a 16th grid becomes eight
 * eighth notes instead of eight note/rest pairs, which is what a drummer would
 * actually put on paper.
 *
 * Duration belongs to the *beat*, not the note, so the gap is measured to the
 * next slot carrying any hit in any lane, not per lane.
 */
function renderBar(bar, res, timeSignature, articulations) {
  const { duration: slotDen, tuplet } = slotValue(res, timeSignature);
  const out = [];

  // Triplet grids are emitted one slot at a time. Merging inside a tuplet
  // produces quarter-note triplets and similar, which need bracket handling
  // this does not do -- and getting that subtly wrong is worse than verbose.
  if (tuplet) {
    const tu = `{tu ${tuplet}}`;
    for (let slot = 0; slot < res; slot++) {
      const hits = hitsAt(bar, slot, articulations);
      out.push(hits.length === 0 ? `r.${slotDen}${tu}`
        : hits.length === 1 ? `${hits[0]}.${slotDen}${tu}`
        : `(${hits.join(' ')}).${slotDen}${tu}`);
    }
    return out.join(' ');
  }

  const values = noteValues(slotDen);
  const events = [];
  for (let slot = 0; slot < res; slot++) {
    const hits = hitsAt(bar, slot, articulations);
    if (hits.length) events.push({ slot, hits });
  }

  const emitRests = (slots) => {
    for (const v of cover(slots, values)) out.push(`r.${v.duration}${v.dots}`);
  };

  if (events.length === 0) {
    emitRests(res);
    return out.join(' ');
  }

  if (events[0].slot > 0) emitRests(events[0].slot);

  events.forEach((event, i) => {
    const next = i + 1 < events.length ? events[i + 1].slot : res;
    const span = next - event.slot;
    const [held, ...remainder] = cover(span, values);

    const body = event.hits.length === 1 ? event.hits[0] : `(${event.hits.join(' ')})`;
    out.push(`${body}.${held.duration}${held.dots}`);

    // Anything the note value could not absorb becomes rests.
    for (const v of remainder) out.push(`r.${v.duration}${v.dots}`);
  });

  return out.join(' ');
}

/**
 * Build a complete AlphaTex document from a chart.
 *
 * `\clef neutral` is what puts drums on a percussion staff rather than a pitched
 * one, and `\articulation defaults` gives us alphaTab's standard kit names.
 */
export function chartToAlphaTex(chart, articulations = DEFAULT_ARTICULATIONS) {
  const ts = chart.time_signature || [4, 4];

  // Metadata arguments go in parentheses. Without them alphaTab still parses,
  // but emits a diagnostic on every load.
  const lines = [`\\title("${escapeTex(chart.title || 'Untitled')}")`];
  if (chart.artist) lines.push(`\\subtitle("${escapeTex(chart.artist)}")`);
  lines.push(
    `\\tempo(${chart.tempo || 120})`,
    '\\track("Drums")',
    '\\instrument(percussion)',
    '\\clef(neutral)',
    '\\articulation(defaults)',
    `\\ts(${ts[0]} ${ts[1]})`,
  );

  const bars = chart.bars || [];
  let currentSection = null;

  for (const bar of bars) {
    const parts = [];
    if (bar.section && bar.section !== currentSection) {
      parts.push(`\\section("${escapeTex(bar.section)}")`);
      currentSection = bar.section;
    }
    parts.push(renderBar(bar, chart.res, ts, articulations));
    lines.push(`${parts.join(' ')} |`);
  }

  if (bars.length === 0) {
    lines.push(`${Array(chart.res).fill(`r.${slotDuration(chart.res, ts)}`).join(' ')} |`);
  }

  return lines.join('\n');
}

function escapeTex(text) {
  return String(text).replace(/"/g, '\\"');
}

// Sync points are not emitted into AlphaTex. They are attached to the score
// model as SyncPoint automations after loading -- see player.js. Keeping them
// out of the text avoids a second source of truth for the same timing data.
