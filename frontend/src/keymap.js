/**
 * Keyboard as drum kit.
 *
 * The layout follows the kit, not the alphabet: kick on the space bar because
 * it is the foot, hats and cymbals on the top row because they sit high, snare
 * and toms on the home row under the fingers. Two hands land naturally on a
 * groove without looking down, which matters when you are playing along to a
 * record rather than typing.
 *
 * Bindings are stored per browser so the layout survives a reload.
 */

const STORAGE_KEY = 'dcdc.keymap.v1';

/** Default bindings: lane key -> array of KeyboardEvent.code values. */
export const DEFAULT_KEYMAP = {
  bd: ['Space'],                        // the foot
  hf: ['KeyZ'],                         // hi-hat pedal, the other foot
  sd: ['KeyF', 'KeyJ'],                 // both hands, for doubles and rolls
  hh: ['KeyD', 'KeyK'],
  ho: ['KeyE', 'KeyI'],                 // open hat
  rd: ['KeyL'],
  cc: ['KeyO'],
  ht: ['KeyR'],
  mt: ['KeyG'],
  lt: ['KeyV'],
};

/** Human-readable label for a KeyboardEvent.code. */
export function keyLabel(code) {
  if (code === 'Space') return 'Space';
  if (code.startsWith('Key')) return code.slice(3);
  if (code.startsWith('Digit')) return code.slice(5);
  if (code.startsWith('Numpad')) return `Num ${code.slice(6)}`;
  return code;
}

export function loadKeymap() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
    if (stored && typeof stored === 'object') return { ...DEFAULT_KEYMAP, ...stored };
  } catch { /* corrupt storage is not worth failing over */ }
  return { ...DEFAULT_KEYMAP };
}

export function saveKeymap(keymap) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(keymap));
  } catch { /* private browsing; bindings just will not persist */ }
}

export function resetKeymap() {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch { /* nothing to clear */ }
  return { ...DEFAULT_KEYMAP };
}

/**
 * Invert a keymap into code -> lane for O(1) lookup while playing.
 *
 * Later bindings win, so a key assigned to two lanes resolves to one rather
 * than firing both.
 */
export function toLookup(keymap) {
  const lookup = {};
  for (const [lane, codes] of Object.entries(keymap)) {
    for (const code of codes || []) lookup[code] = lane;
  }
  return lookup;
}

/** Which lane, if any, a binding would collide with. */
export function findConflict(keymap, code, exceptLane) {
  for (const [lane, codes] of Object.entries(keymap)) {
    if (lane !== exceptLane && (codes || []).includes(code)) return lane;
  }
  return null;
}
