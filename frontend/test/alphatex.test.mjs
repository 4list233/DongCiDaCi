/**
 * Parse-checks the AlphaTex emitter against alphaTab itself.
 *
 * This exists because alphaTab validates percussion articulation names against
 * a fixed vocabulary and rejects the *entire score* if one is wrong. A plausible
 * but invented name (CrashHigh instead of CrashHighHit) produces a blank page,
 * not a missing note -- so the names get checked by the real parser rather than
 * trusted.
 *
 *     cd frontend && npm test
 */

import * as alphaTab from '@coderline/alphatab';
import { chartToAlphaTex, DEFAULT_ARTICULATIONS } from '../src/alphatex.js';

let failures = 0;

function check(name, condition, detail = '') {
  if (condition) {
    console.log(`  ok   ${name}`);
  } else {
    failures++;
    console.log(`  FAIL ${name}${detail ? ` -- ${detail}` : ''}`);
  }
}

function parse(chart) {
  const tex = chartToAlphaTex(chart);
  const importer = new alphaTab.importer.AlphaTexImporter();
  importer.initFromString(tex, new alphaTab.Settings());
  const score = importer.readScore();

  const diagnostics = [
    ...importer.lexerDiagnostics.items,
    ...importer.parserDiagnostics.items,
  ];
  return { tex, score, diagnostics };
}

function chartWith(lanes, res = 16) {
  return {
    title: 'Test', artist: 'Nobody', res, time_signature: [4, 4], tempo: 120,
    bars: [{ n: 1, lanes }],
    sync: [{ bar: 1, time: 0, tempo: 120 }],
  };
}

// --- every lane in the vocabulary must be a real articulation ---------------

console.log('\nall lanes parse');
{
  // One lane per bar, so a single bad name cannot be masked by the others.
  const laneKeys = Object.keys(DEFAULT_ARTICULATIONS);
  for (const key of laneKeys) {
    const chart = chartWith({ [key]: 'o---------------' });
    try {
      const { score } = parse(chart);
      const notes = score.tracks[0].staves[0].bars[0].voices[0].beats
        .reduce((n, b) => n + b.notes.length, 0);
      check(`${key} (${DEFAULT_ARTICULATIONS[key]})`, notes === 1, `got ${notes} notes`);
    } catch (err) {
      check(`${key} (${DEFAULT_ARTICULATIONS[key]})`, false, String(err.message || err));
    }
  }
}

// --- a realistic chart round-trips ------------------------------------------

console.log('\nrealistic groove');
{
  const chart = {
    title: 'Test Groove', artist: 'Nobody', res: 16, time_signature: [4, 4], tempo: 120,
    bars: [
      { n: 1, section: 'verse', lanes: {
        hh: 'x-x-x-x-x-x-x-x-', sd: '----o---g---o---', bd: 'o--o----o-------' } },
      { n: 2, lanes: {
        cc: 'x---------------', rd: '--x-x-x-x-x-x-x-', sd: '----O-------f---',
        bd: 'o-------o-------', mt: '------------o---' } },
    ],
    sync: [{ bar: 1, time: 0, tempo: 120 }],
  };

  const { score, diagnostics } = parse(chart);
  const staff = score.tracks[0].staves[0];

  check('title survives', score.title === 'Test Groove');
  check('two master bars', score.masterBars.length === 2);
  check('track is percussion', staff.isPercussion === true);
  check('no parser diagnostics', diagnostics.length === 0,
        diagnostics.map((d) => d.message).join('; '));

  const bar1Notes = staff.bars[0].voices[0].beats.reduce((n, b) => n + b.notes.length, 0);
  check('bar 1 note count', bar1Notes === 14, `got ${bar1Notes}, expected 14`);

  const bar2Notes = staff.bars[1].voices[0].beats.reduce((n, b) => n + b.notes.length, 0);
  check('bar 2 note count', bar2Notes === 13, `got ${bar2Notes}, expected 13`);

  // Simultaneous hits must land in one beat, not sequential ones, or the
  // notation shows a kick *after* the hi-hat instead of under it.
  const firstBeat = staff.bars[0].voices[0].beats.find((b) => b.notes.length);
  check('simultaneous hits share a beat', firstBeat.notes.length === 2,
        `got ${firstBeat.notes.length}`);

  // Distinct lanes must resolve to distinct instruments.
  const articulations = new Set(
    staff.bars[1].voices[0].beats.flatMap((b) => b.notes.map((n) => n.percussionArticulation)),
  );
  check('bar 2 uses 5 distinct voices', articulations.size === 5,
        `got ${articulations.size}: ${[...articulations].join(',')}`);
}

// --- dynamics reach the model ------------------------------------------------

console.log('\ndynamics and rudiments');
{
  const { score } = parse(chartWith({ sd: 'O---o---g---f---' }));
  const beats = score.tracks[0].staves[0].bars[0].voices[0].beats.filter((b) => b.notes.length);

  check('four snare events', beats.length === 4, `got ${beats.length}`);
  check('accent is marked', beats[0].notes[0].isEffectSlurOrigin === false
        && beats[0].notes[0].accentuated !== undefined,
        'accentuated property missing');
  check('accent differs from plain hit',
        beats[0].notes[0].accentuated !== beats[1].notes[0].accentuated,
        `accent=${beats[0].notes[0].accentuated} plain=${beats[1].notes[0].accentuated}`);
  check('ghost is marked',
        beats[2].notes[0].isGhost === true,
        `isGhost=${beats[2].notes[0].isGhost}`);
}

// --- note lengths -------------------------------------------------------------

console.log('\nnote lengths');
{
  // A plain eighth-note hi-hat on a 16-slot grid must come out as eight eighth
  // notes, not eight note/rest pairs. This is the single biggest readability
  // difference between a usable chart and an unusable one.
  const { score } = parse(chartWith({ hh: 'x-x-x-x-x-x-x-x-' }));
  const beats = score.tracks[0].staves[0].bars[0].voices[0].beats;

  check('eighths do not become note+rest pairs', beats.length === 8, `got ${beats.length} beats`);
  check('every beat sounds', beats.every((b) => b.notes.length === 1));
  check('written as eighth notes', beats.every((b) => b.duration === 8),
        `durations: ${beats.map((b) => b.duration).join(',')}`);
}
{
  // Notes hold until the next event in any lane, so a 16th-note kick shortens
  // only the beat it interrupts.
  const { score } = parse(chartWith({ hh: 'x-x-x-x-x-x-x-x-', bd: 'o--o----o-------' }));
  const beats = score.tracks[0].staves[0].bars[0].voices[0].beats;
  const durations = beats.map((b) => b.duration).join(',');

  check('gap to next event sets the value', durations === '8,16,16,8,8,8,8,8,8',
        `got ${durations}`);
  check('all nine events sound', beats.every((b) => b.notes.length >= 1));
}
{
  const { score } = parse(chartWith({ bd: 'o-------o-------' }));
  const beats = score.tracks[0].staves[0].bars[0].voices[0].beats;
  check('a half-bar gap is a half note', beats.length === 2 && beats.every((b) => b.duration === 2),
        `got ${beats.map((b) => b.duration).join(',')}`);
}
{
  const { score } = parse(chartWith({ sd: 'o-----o---------' }));
  const beats = score.tracks[0].staves[0].bars[0].voices[0].beats;
  check('a 6-slot gap is a dotted quarter', beats[0].duration === 4 && beats[0].dots === 1,
        `duration=${beats[0].duration} dots=${beats[0].dots}`);
}
{
  const { score } = parse(chartWith({ sd: '----o-----------' }));
  const beats = score.tracks[0].staves[0].bars[0].voices[0].beats;
  check('a leading gap becomes a rest', beats[0].notes.length === 0,
        'first beat should be a rest');
  check('then the hit', beats.some((b) => b.notes.length === 1));
}
{
  const { score } = parse(chartWith({}));
  const beats = score.tracks[0].staves[0].bars[0].voices[0].beats;
  check('an empty bar is rests only', beats.every((b) => b.notes.length === 0));
}

// --- resolutions --------------------------------------------------------------

console.log('\nresolutions');
for (const [res, label] of [[8, 'eighths'], [12, 'eighth triplets'], [16, 'sixteenths'], [32, 'thirty-seconds']]) {
  const chart = chartWith({ hh: 'x'.repeat(res) }, res);
  try {
    const { score } = parse(chart);
    const notes = score.tracks[0].staves[0].bars[0].voices[0].beats
      .reduce((n, b) => n + b.notes.length, 0);
    check(`res ${res} (${label})`, notes === res, `got ${notes} notes, expected ${res}`);
  } catch (err) {
    check(`res ${res} (${label})`, false, String(err.message || err));
  }
}

console.log(failures === 0 ? '\nall alphatex checks passed\n' : `\n${failures} FAILED\n`);
process.exit(failures === 0 ? 0 : 1);
