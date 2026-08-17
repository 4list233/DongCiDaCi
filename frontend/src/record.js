/**
 * Playing a part in, rather than clicking it in.
 *
 * Two things make this work at all:
 *
 * Timing. alphaTab reports playback position on an interval, not per frame, so
 * using the last reported value as the time of a keystroke quantises every hit
 * to that interval before the real quantiser ever sees it. Instead each
 * position report is paired with a `performance.now()` reading, and a keystroke
 * is timed by extrapolating from that pair. The error becomes the age of the
 * last report rather than the whole interval.
 *
 * Latency. There is real delay between hitting a key and the browser
 * dispatching the event, and more between a sample being scheduled and heard.
 * A player compensates by anticipating, so their hits land consistently early
 * or late -- consistently being the useful part, because a single offset
 * corrects it. `latencyMs` is that offset, and it is worth calibrating once.
 */

import { toLookup } from './keymap.js';

/** How stale a position report can be before a hit is not trusted. */
const STALE_POSITION_MS = 400;

export class Recorder {
  /**
   * @param {object} options
   * @param {() => object} options.getChart      current chart, for the grid
   * @param {(hit) => void} options.onHit        a captured, quantised hit
   * @param {(state) => void} [options.onState]  armed/counting/recording changes
   */
  constructor({ getChart, onHit, onState, onIgnored } = {}) {
    this.getChart = getChart;
    this.onHit = onHit || (() => {});
    this.onState = onState || (() => {});
    // A key pressed with nowhere to put it. Silence here means playing a
    // whole take into the void while the button still says Recording.
    this.onIgnored = onIgnored || (() => {});

    this.keymap = {};
    this.lookup = {};
    this.latencyMs = 0;
    this.armed = false;
    this.replaceLanes = false;   // overwrite a lane rather than layering onto it

    // The most recent (playback ms, performance.now() ms) pair.
    this._anchor = null;
    // Keys currently held, so auto-repeat does not become a drum roll.
    this._held = new Set();
    this._onKeyDown = this._onKeyDown.bind(this);
    this._onKeyUp = this._onKeyUp.bind(this);
  }

  setKeymap(keymap) {
    this.keymap = keymap;
    this.lookup = toLookup(keymap);
  }

  setLatency(ms) {
    this.latencyMs = Number(ms) || 0;
  }

  /** Feed alphaTab's position reports in, paired with a local clock reading. */
  updatePosition(playbackMs) {
    this._anchor = { playbackMs, at: performance.now() };
  }

  /** Playback position now, extrapolated past the last report. */
  currentPlaybackMs() {
    if (!this._anchor) return null;
    const age = performance.now() - this._anchor.at;
    if (age > STALE_POSITION_MS) return null;   // playback stopped or stalled
    return this._anchor.playbackMs + age;
  }

  arm() {
    if (this.armed) return;
    this.armed = true;
    this._held.clear();
    window.addEventListener('keydown', this._onKeyDown);
    window.addEventListener('keyup', this._onKeyUp);
    this.onState({ armed: true });
  }

  disarm() {
    if (!this.armed) return;
    this.armed = false;
    this._held.clear();
    window.removeEventListener('keydown', this._onKeyDown);
    window.removeEventListener('keyup', this._onKeyUp);
    this.onState({ armed: false });
  }

  _onKeyUp(event) {
    this._held.delete(event.code);
  }

  _onKeyDown(event) {
    // Never steal keys from a text field, and leave browser shortcuts alone.
    if (isTyping(event.target) || event.metaKey || event.ctrlKey || event.altKey) return;

    const lane = this.lookup[event.code];
    if (!lane) return;

    // Holding a key must not machine-gun the lane. A real double stroke means
    // releasing and hitting again.
    event.preventDefault();
    if (event.repeat || this._held.has(event.code)) return;
    this._held.add(event.code);

    const playbackMs = this.currentPlaybackMs();
    if (playbackMs == null) {
      this.onIgnored('notplaying');
      return;
    }

    const seconds = (playbackMs - this.latencyMs) / 1000;
    const position = locate(this.getChart(), seconds);
    if (!position) {
      this.onIgnored('offchart');
      return;
    }

    this.onHit({ lane, ...position, seconds });
  }
}

/**
 * Which bar and slot a moment in the recording lands on.
 *
 * Bars are located from the chart's sync points, which come from the beat
 * tracker and therefore follow a performance that drifts. Within a bar the
 * grid is assumed even, which is true enough at one bar's width even when the
 * tempo is moving.
 */
export function locate(chart, seconds) {
  const sync = chart?.sync;
  if (!sync?.length || !chart?.bars?.length) return null;

  let index = -1;
  for (let i = 0; i < sync.length; i += 1) {
    if (sync[i].time <= seconds) index = i;
    else break;
  }
  if (index < 0) return null;                 // before the first downbeat

  const start = sync[index];
  const next = sync[index + 1];
  const bar = barNumbered(chart, start.bar);
  if (!bar) return null;

  // From the chart, not from a lane string. Inferring it from the patterns meant
  // an empty bar had no resolution and could not be recorded into -- which is
  // precisely the bar you most want to play a part into.
  const res = chart.res || resolutionOf(bar);
  if (!res) return null;

  // Without a following sync point (the last bar) fall back to this bar's own
  // duration implied by the previous gap, so the final bar stays recordable.
  const previous = sync[index - 1];
  const span = next
    ? next.time - start.time
    : previous ? start.time - previous.time : null;
  if (!span || span <= 0) return null;

  const ratio = (seconds - start.time) / span;
  let slot = Math.round(ratio * res);

  // A hit a hair before the barline belongs to the next bar's downbeat, which
  // is exactly where a crash lands.
  if (slot >= res) {
    const following = barNumbered(chart, start.bar + 1);
    if (!following) return { bar: start.bar, slot: res - 1, res };
    return { bar: following.n ?? following.index, slot: 0, res };
  }
  if (slot < 0) slot = 0;

  return { bar: start.bar, slot, res };
}

/**
 * A bar by its number. Charts from the API number bars with `n`; the field was
 * read as `index` here, which happened to work only because of the positional
 * fallback, and would have silently mis-targeted any chart with a gap.
 */
function barNumbered(chart, number) {
  return chart.bars.find((b) => (b.n ?? b.index) === number) || chart.bars[number - 1];
}

/** Fallback resolution when a chart does not carry one. */
function resolutionOf(bar) {
  const lanes = bar?.lanes;
  if (!lanes) return null;
  for (const value of Object.values(lanes)) {
    if (typeof value === 'string' && value.length) return value.length;
  }
  return null;
}

function isTyping(element) {
  if (!element) return false;
  const tag = element.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || element.isContentEditable;
}

/**
 * A count-in before recording starts.
 *
 * Playing in cold means the first bar is always wrong. Counting in at the
 * song's own tempo gives you somewhere to pick the time up from.
 */
export class CountIn {
  constructor(audioContext) {
    this.context = audioContext;
    this.timers = [];
  }

  /**
   * Click `beats` times at `bpm`, resolving when the last click has sounded.
   * The downbeat is pitched higher so you know where bar one starts.
   */
  run(bpm, beats = 4, beatsPerBar = 4) {
    this.cancel();
    const interval = 60000 / (bpm || 120);

    return new Promise((resolve) => {
      for (let i = 0; i < beats; i += 1) {
        this.timers.push(setTimeout(() => {
          this._click(i % beatsPerBar === 0);
          if (i === beats - 1) {
            this.timers.push(setTimeout(resolve, interval));
          }
        }, i * interval));
      }
    });
  }

  cancel() {
    for (const timer of this.timers) clearTimeout(timer);
    this.timers = [];
  }

  _click(accent) {
    const context = this.context;
    if (!context) return;
    const now = context.currentTime;

    const oscillator = context.createOscillator();
    const gain = context.createGain();
    oscillator.frequency.value = accent ? 1600 : 900;
    gain.gain.setValueAtTime(0.28, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.06);

    oscillator.connect(gain).connect(context.destination);
    oscillator.start(now);
    oscillator.stop(now + 0.07);
  }
}
