/**
 * The controls for playing a part in, and for editing by clicking the score.
 *
 * Kept apart from main.js because this is a mode rather than a control: while
 * armed, the keyboard belongs to the drum kit, and the rules about what a
 * keystroke means change entirely.
 */

import { Recorder, CountIn, locate } from './record.js';
import {
  DEFAULT_KEYMAP, loadKeymap, saveKeymap, resetKeymap, keyLabel, findConflict,
} from './keymap.js';

const LATENCY_KEY = 'dcdc.latency';
const COUNTIN_KEY = 'dcdc.countin';

export class RecordUI {
  /**
   * @param {object} deps
   * @param {() => object} deps.getChart
   * @param {(bar, lane, slot, char) => boolean} deps.applyHit
   * @param {object} deps.player
   * @param {Array} deps.lanes           vocabulary, for labelling the bindings
   * @param {(text) => void} deps.setStatus
   */
  constructor({ getChart, applyHit, player, lanes, setStatus }) {
    this.getChart = getChart;
    this.applyHit = applyHit;
    this.player = player;
    this.lanes = lanes;
    this.setStatus = setStatus;

    this.keymap = loadKeymap();
    this.latency = Number(localStorage.getItem(LATENCY_KEY)) || 0;
    this.countInBeats = Number(localStorage.getItem(COUNTIN_KEY) ?? 4);
    this.rebinding = null;

    this.recorder = new Recorder({
      getChart,
      onHit: (hit) => this._onHit(hit),
      onIgnored: (why) => this._onIgnored(why),
    });
    this.recorder.setKeymap(this.keymap);
    this.recorder.setLatency(this.latency);

    this.countIn = new CountIn(this._audioContext());
  }

  _audioContext() {
    try {
      return new (window.AudioContext || window.webkitAudioContext)();
    } catch {
      return null;      // no audio context: the count-in is silent, not fatal
    }
  }

  // --- recording ------------------------------------------------------------

  get armed() { return this.recorder.armed; }

  async toggle() {
    if (this.recorder.armed) {
      this.disarm();
      return;
    }
    await this.arm();
  }

  async arm() {
    if (!this.getChart()) {
      this.setStatus('Transcribe a song first — there is nothing to play into.');
      return;
    }

    const button = document.querySelector('#record');
    button?.classList.add('counting');

    if (this.countInBeats > 0) {
      this.setStatus(`Counting in ${this.countInBeats} beats…`);
      const tempo = this.getChart()?.tempo || 120;
      await this.countIn.run(tempo, this.countInBeats);
    }

    button?.classList.remove('counting');
    this.recorder.arm();
    this._reflect();
    this.setStatus('Recording — play along. Your hits land on the grid as you play.');

    // Recording against silence records nothing useful, so start playback too,
    // and say so if it did not start -- otherwise every keystroke is discarded
    // while the button still reads Recording.
    this.player?.playPause();
    setTimeout(() => {
      if (this.recorder.armed && this.recorder.currentPlaybackMs() == null) {
        this.setStatus(
          'Armed, but nothing is playing — press Play. Hits are only recorded '
          + 'against a running playhead.'
        );
      }
    }, 700);
  }

  disarm() {
    this.countIn.cancel();
    this.recorder.disarm();
    this._reflect();
    this.setStatus('Stopped recording.');
  }

  /** Feed playback position through, so hits can be timed between reports. */
  updatePosition(playbackMs) {
    this.recorder.updatePosition(playbackMs);
  }

  _onIgnored(why) {
    this.setStatus(why === 'notplaying'
      ? 'Not recorded — nothing is playing. Press Play, then play along.'
      : 'Not recorded — the playhead is outside the chart.');
  }

  _onHit({ lane, bar, slot }) {
    const ok = this.applyHit(bar, lane, slot, 'x');
    if (ok) this._flash(lane);
  }

  /** Brief confirmation that a key registered, since the note may be off-screen. */
  _flash(lane) {
    const button = document.querySelector('#record');
    if (!button) return;
    button.dataset.lastLane = lane;
    button.classList.add('hit');
    clearTimeout(this._flashTimer);
    this._flashTimer = setTimeout(() => button.classList.remove('hit'), 90);
  }

  _reflect() {
    const button = document.querySelector('#record');
    if (!button) return;
    button.classList.toggle('armed', this.recorder.armed);
    button.setAttribute('aria-pressed', String(this.recorder.armed));
    button.textContent = this.recorder.armed ? '● Recording' : '● Record';
  }

  // --- settings -------------------------------------------------------------

  openSettings() {
    this.renderBindings();
    document.querySelector('#latency').value = this.latency;
    document.querySelector('#latencylabel').textContent = `${this.latency} ms`;
    document.querySelector('#countin').value = String(this.countInBeats);
    document.querySelector('#settings')?.showModal();
  }

  renderBindings() {
    const host = document.querySelector('#keybindings');
    if (!host) return;
    host.replaceChildren();

    for (const lane of this.lanes) {
      const row = document.createElement('div');
      row.className = 'binding';

      const label = document.createElement('span');
      label.textContent = lane.label;

      const keys = document.createElement('span');
      keys.className = 'keys';
      for (const code of this.keymap[lane.key] || []) {
        const key = document.createElement('button');
        key.type = 'button';
        key.className = 'keycap';
        key.textContent = keyLabel(code);
        key.addEventListener('click', () => this._startRebind(lane.key, code, key));
        keys.appendChild(key);
      }

      const add = document.createElement('button');
      add.type = 'button';
      add.className = 'keycap add';
      add.textContent = '+';
      add.title = `Add another key for ${lane.label}`;
      add.addEventListener('click', () => this._startRebind(lane.key, null, add));
      keys.appendChild(add);

      row.append(label, keys);
      host.appendChild(row);
    }
  }

  /**
   * Wait for the next keypress and bind it.
   *
   * Capture phase, because the recorder may also be listening and the binding
   * must win. A key already used elsewhere is taken from that lane rather than
   * silently shadowing it, since two lanes sharing a key means one of them can
   * never be played.
   */
  _startRebind(laneKey, replacing, element) {
    if (this.rebinding) this.rebinding();

    element.classList.add('listening');
    element.textContent = '…';

    const onKey = (event) => {
      event.preventDefault();
      event.stopPropagation();
      finish(event.code === 'Escape' ? null : event.code);
    };

    const finish = (code) => {
      window.removeEventListener('keydown', onKey, true);
      this.rebinding = null;
      element.classList.remove('listening');

      if (code) {
        const clash = findConflict(this.keymap, code, laneKey);
        if (clash) {
          this.keymap[clash] = (this.keymap[clash] || []).filter((c) => c !== code);
        }
        const current = this.keymap[laneKey] || [];
        this.keymap[laneKey] = replacing
          ? current.map((c) => (c === replacing ? code : c))
          : [...current, code];

        saveKeymap(this.keymap);
        this.recorder.setKeymap(this.keymap);
      }
      this.renderBindings();
    };

    window.addEventListener('keydown', onKey, true);
    this.rebinding = () => finish(null);
  }

  setLatency(ms) {
    this.latency = Number(ms) || 0;
    this.recorder.setLatency(this.latency);
    localStorage.setItem(LATENCY_KEY, String(this.latency));
    document.querySelector('#latencylabel').textContent = `${this.latency} ms`;
  }

  setCountIn(beats) {
    this.countInBeats = Number(beats) || 0;
    localStorage.setItem(COUNTIN_KEY, String(this.countInBeats));
  }

  resetKeys() {
    this.keymap = resetKeymap();
    this.recorder.setKeymap(this.keymap);
    this.renderBindings();
  }
}

/**
 * Make the score itself the surface you work on.
 *
 * alphaTab renders; it does not edit. But it exposes the beat under a click,
 * and because the AlphaTex is generated here that beat maps straight back to a
 * bar and slot. That is enough to treat the notation as the document rather
 * than as a picture of one, with the grid below it as a fallback instead of
 * the only way in.
 *
 * Clicking has to mean one thing at a time. In `play` it moves the playhead --
 * which is what you want ninety percent of the time, going back over a fill.
 * In `edit` it removes the note you clicked. A mode is honest about that; a
 * modifier key would leave people clicking and getting a surprise.
 */
export class ScoreInteraction {
  constructor({ player, getChart, applyHit, onSeek, setStatus }) {
    this.player = player;
    this.getChart = getChart;
    this.applyHit = applyHit;
    this.onSeek = onSeek || (() => {});
    this.setStatus = setStatus || (() => {});
    this.mode = 'play';

    const api = player?.api;
    if (!api) return;

    api.beatMouseDown?.on((beat) => this._onBeat(beat));
    api.noteMouseDown?.on((note) => this._onNote(note));
  }

  setMode(mode) {
    this.mode = mode;
    document.body.classList.toggle('editing-score', mode === 'edit');
    this.setStatus(mode === 'edit'
      ? 'Edit mode — click a note on the score to remove it.'
      : 'Click anywhere on the score to play from there.');
  }

  _onBeat(beat) {
    const chart = this.getChart();
    if (!chart || this.mode !== 'play') return;

    const position = beatPosition(beat, chart);
    if (!position) return;
    this.onSeek(position.bar);
  }

  _onNote(note) {
    const chart = this.getChart();
    if (!chart || this.mode !== 'edit') return;

    const position = beatPosition(note.beat, chart);
    const lane = laneOfNote(note);
    if (!position || !lane) return;

    this.applyHit(position.bar, lane, position.slot, '-');
    this.setStatus(`removed ${lane} in bar ${position.bar}`);
  }
}

/** Where a beat sits, as a bar number and slot index. */
function beatPosition(beat, chart) {
  const barIndex = beat?.voice?.bar?.index;
  if (barIndex == null) return null;

  const bar = chart.bars?.[barIndex];
  if (!bar) return null;

  // A bar is one whole note of ticks; slots divide it evenly.
  const TICKS_PER_BAR = 3840;
  const within = (beat.playbackStart ?? 0) % TICKS_PER_BAR;
  const slot = Math.round((within / TICKS_PER_BAR) * chart.res);

  return { bar: bar.n ?? barIndex + 1, slot: Math.min(slot, chart.res - 1) };
}

/** Which lane a rendered note belongs to, by its percussion MIDI value. */
function laneOfNote(note) {
  const midi = note?.percussionArticulation ?? note?.value;
  return MIDI_TO_LANE[midi] ?? null;
}

// Mirrors chart.LANES on the backend. Only used to read a click back.
const MIDI_TO_LANE = {
  49: 'cc', 51: 'rd', 42: 'hh', 48: 'ht', 45: 'mt', 41: 'lt',
  38: 'sd', 36: 'bd', 44: 'hf', 46: 'ho',
};
