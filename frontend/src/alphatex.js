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
 * Render one bar as AlphaTex beats.
 *
 * AlphaTex wants a duration on each beat and simultaneous hits grouped in
 * parentheses. Rests are emitted explicitly rather than by extending the
 * previous note, because a drum chart that hides its rests is unreadable.
 */
function renderBar(bar, res, timeSignature, articulations) {
  const { duration, tuplet } = slotValue(res, timeSignature);
  // Beat effect, applied after the duration: `SnareHit.8{tu 3}`.
  const tu = tuplet ? `{tu ${tuplet}}` : '';
  const out = [];

  for (let slot = 0; slot < res; slot++) {
    const hits = [];
    let isOpen = false;

    for (const [lane, pattern] of Object.entries(bar.lanes || {})) {
      const ch = pattern[slot];
      if (!ch || ch === REST) continue;

      let name = articulations[lane];
      if (!name) continue;
      if (lane === 'hh' && OPEN_CHARS.has(ch)) {
        name = 'HiHatOpen';
        isOpen = true;
      }

      // Accents and ghosts are effects on the note, not different notes.
      let effects = '';
      if (ch === 'X' || ch === 'O') effects = '{ac}';
      else if (ch === 'g') effects = '{g}';
      else if (ch === 'f') effects = '{gr}';

      hits.push(`${name}${effects}`);
    }

    if (hits.length === 0) {
      out.push(`r.${duration}${tu}`);
    } else if (hits.length === 1) {
      out.push(`${hits[0]}.${duration}${tu}`);
    } else {
      out.push(`(${hits.join(' ')}).${duration}${tu}`);
    }
    void isOpen;
  }

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
