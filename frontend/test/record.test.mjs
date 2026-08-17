/**
 * Checks the mapping from "a key was hit at this moment" to "this bar, this
 * slot".
 *
 * This is the part of playing a part in that has to be right. A hit landing one
 * slot late is worse than no hit at all, because it looks deliberate.
 *
 *     cd frontend && npm test
 */

import { locate } from '../src/record.js';
import { DEFAULT_KEYMAP, toLookup, findConflict, keyLabel } from '../src/keymap.js';

let failures = 0;

function check(name, condition, detail = '') {
  if (condition) {
    console.log(`  ok   ${name}`);
  } else {
    failures++;
    console.log(`  FAIL ${name}${detail ? ` -- ${detail}` : ''}`);
  }
}

/** Four bars of 16ths at 120bpm, so a bar is 2s and a slot is 125ms. */
function chart({ bars = 4, res = 16, bpm = 120 } = {}) {
  const barSeconds = (60 / bpm) * 4;
  return {
    bars: Array.from({ length: bars }, (_, i) => ({
      index: i + 1,
      lanes: { bd: '-'.repeat(res), sd: '-'.repeat(res) },
    })),
    sync: Array.from({ length: bars }, (_, i) => ({
      bar: i + 1,
      time: i * barSeconds,
    })),
  };
}

console.log('locate: time -> bar and slot');
{
  const c = chart();

  const downbeat = locate(c, 0);
  check('the downbeat is bar 1 slot 0',
    downbeat?.bar === 1 && downbeat?.slot === 0, JSON.stringify(downbeat));

  const beatTwo = locate(c, 0.5);
  check('beat 2 of bar 1 is slot 4',
    beatTwo?.bar === 1 && beatTwo?.slot === 4, JSON.stringify(beatTwo));

  const barTwo = locate(c, 2.0);
  check('the second downbeat is bar 2 slot 0',
    barTwo?.bar === 2 && barTwo?.slot === 0, JSON.stringify(barTwo));

  const lastSlot = locate(c, 1.875);
  check('the last sixteenth of bar 1 is slot 15',
    lastSlot?.bar === 1 && lastSlot?.slot === 15, JSON.stringify(lastSlot));

  check('resolution is reported', locate(c, 0)?.res === 16);
}

console.log('locate: rounding to the nearest slot');
{
  const c = chart();

  const early = locate(c, 0.49);          // 10ms early for beat 2
  check('a hit slightly early still lands on the beat',
    early?.slot === 4, JSON.stringify(early));

  const late = locate(c, 0.51);           // 10ms late
  check('a hit slightly late still lands on the beat',
    late?.slot === 4, JSON.stringify(late));

  const between = locate(c, 0.5625);      // exactly between slots 4 and 5
  check('a hit halfway between slots picks one',
    between?.slot === 4 || between?.slot === 5, JSON.stringify(between));
}

console.log('locate: the barline');
{
  const c = chart();

  // A crash is played fractionally before the downbeat it belongs to.
  const anticipated = locate(c, 1.99);
  check('a hit just before the barline belongs to the next downbeat',
    anticipated?.bar === 2 && anticipated?.slot === 0, JSON.stringify(anticipated));

  const past = locate(c, 7.99);
  check('a hit before the end of the last bar stays in it',
    past?.bar === 4, JSON.stringify(past));

  check('a hit before the first downbeat is refused', locate(c, -0.5) === null);
}

console.log('locate: drifting tempo');
{
  // Sync points from a real beat tracker are not evenly spaced. Bar 2 is
  // stretched, so its slots must stretch with it rather than staying at 125ms.
  const c = {
    bars: [
      { index: 1, lanes: { bd: '-'.repeat(16) } },
      { index: 2, lanes: { bd: '-'.repeat(16) } },
      { index: 3, lanes: { bd: '-'.repeat(16) } },
    ],
    sync: [{ bar: 1, time: 0 }, { bar: 2, time: 2.0 }, { bar: 3, time: 5.0 }],
  };

  const midway = locate(c, 3.5);   // half way through a 3s bar
  check('slots stretch with a slower bar',
    midway?.bar === 2 && midway?.slot === 8, JSON.stringify(midway));

  const quarter = locate(c, 2.75);
  check('a quarter into the slow bar is slot 4',
    quarter?.bar === 2 && quarter?.slot === 4, JSON.stringify(quarter));
}

console.log('locate: bad input');
{
  check('no chart', locate(null, 1) === null);
  check('no sync points', locate({ bars: [{ index: 1, lanes: {} }], sync: [] }, 1) === null);
  check('no bars', locate({ bars: [], sync: [{ bar: 1, time: 0 }] }, 1) === null);
  check('a bar with no lanes yields nothing',
    locate({ bars: [{ index: 1, lanes: {} }], sync: [{ bar: 1, time: 0 }, { bar: 2, time: 2 }] }, 0.5) === null);
}

console.log('keymap');
{
  const lookup = toLookup(DEFAULT_KEYMAP);
  check('space is the kick', lookup.Space === 'bd');
  check('both hands reach the snare',
    lookup.KeyF === 'sd' && lookup.KeyJ === 'sd');

  const codes = Object.values(DEFAULT_KEYMAP).flat();
  check('no default binding is used twice',
    new Set(codes).size === codes.length,
    codes.filter((c, i) => codes.indexOf(c) !== i).join(', '));

  check('a conflict is found', findConflict(DEFAULT_KEYMAP, 'Space', 'sd') === 'bd');
  check('rebinding a lane onto its own key is not a conflict',
    findConflict(DEFAULT_KEYMAP, 'Space', 'bd') === null);

  check('labels are readable',
    keyLabel('KeyF') === 'F' && keyLabel('Space') === 'Space');
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall record checks passed');
process.exit(failures ? 1 : 0);
