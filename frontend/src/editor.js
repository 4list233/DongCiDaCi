/**
 * The grid editor.
 *
 * Drums are a grid, not a score: a fixed set of voices against a fixed set of
 * subdivisions. That is why this is a step sequencer rather than a general
 * notation editor -- it is the right interface for the instrument, and it is an
 * order of magnitude less work than click-on-staff editing.
 *
 * Editing mutates chart.json directly. AlphaTex is regenerated from it for
 * display, never parsed back.
 */

const REST = '-';

// Clicking a cell cycles through the articulations that make sense for that
// voice. Cymbals do not get ghost notes; only the snare does.
const CYCLES = {
  cymbal: [REST, 'x', 'X', '+'],
  snare: [REST, 'o', 'O', 'g', 'f'],
  drum: [REST, 'o', 'O', 'f'],
};

const CYMBAL_LANES = new Set(['cc', 'rd', 'hh', 'hf']);

function cycleFor(laneKey) {
  if (laneKey === 'sd') return CYCLES.snare;
  if (CYMBAL_LANES.has(laneKey)) return CYCLES.cymbal;
  return CYCLES.drum;
}

export class GridEditor {
  /**
   * @param {HTMLElement} host
   * @param {{lanes: Array, onChange: Function, onSeek: Function}} opts
   */
  constructor(host, { lanes, onChange, onSeek }) {
    this.host = host;
    this.lanes = lanes;
    this.onChange = onChange || (() => {});
    this.onSeek = onSeek || (() => {});
    this.chart = null;
    this.activeBar = null;
    this._painting = null;

    // Drag-paint: mousedown sets a value, mouseover applies it while held.
    // Programming sixteen hi-hats one click at a time is miserable.
    document.addEventListener('mouseup', () => { this._painting = null; });
  }

  setChart(chart) {
    this.chart = chart;
    this.render();
  }

  /** Highlight the bar currently under the playback cursor. */
  setActiveBar(barNumber) {
    if (this.activeBar === barNumber) return;
    this.activeBar = barNumber;
    for (const el of this.host.querySelectorAll('.bar')) {
      el.classList.toggle('playing', Number(el.dataset.bar) === barNumber);
    }
    const el = this.host.querySelector(`.bar[data-bar="${barNumber}"]`);
    if (el) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  render() {
    this.host.replaceChildren();
    if (!this.chart) return;

    const { res } = this.chart;
    const slotsPerBeat = res / this.chart.time_signature[0];

    for (const bar of this.chart.bars) {
      this.host.appendChild(this._renderBar(bar, res, slotsPerBeat));
    }
  }

  _renderBar(bar, res, slotsPerBeat) {
    const wrap = document.createElement('section');
    wrap.className = 'bar';
    wrap.dataset.bar = String(bar.n);
    if (bar.n === this.activeBar) wrap.classList.add('playing');

    // --- header: number, section, jump-to-audio, per-bar note
    const head = document.createElement('header');

    const num = document.createElement('button');
    num.className = 'barnum';
    num.textContent = bar.n;
    num.title = 'Play from this bar';
    num.addEventListener('click', () => this.onSeek(bar.n));
    head.appendChild(num);

    const section = document.createElement('input');
    section.className = 'section';
    section.placeholder = 'section';
    section.value = bar.section || '';
    section.addEventListener('change', () => {
      bar.section = section.value.trim() || null;
      this._changed();
    });
    head.appendChild(section);

    const note = document.createElement('input');
    note.className = 'note';
    note.placeholder = 'what the machine missed…';
    note.value = bar.note || '';
    note.addEventListener('change', () => {
      bar.note = note.value.trim() || null;
      this._changed();
    });
    head.appendChild(note);

    const dup = document.createElement('button');
    dup.className = 'ghostbtn';
    dup.textContent = 'copy →';
    dup.title = 'Copy this bar over the next one';
    dup.addEventListener('click', () => this._copyToNext(bar));
    head.appendChild(dup);

    wrap.appendChild(head);

    // --- the grid itself
    const grid = document.createElement('div');
    grid.className = 'grid';
    grid.style.setProperty('--slots', String(res));

    for (const lane of this.lanes) {
      const label = document.createElement('div');
      label.className = 'lanelabel';
      label.textContent = lane.label;
      label.title = lane.key;
      grid.appendChild(label);

      const row = document.createElement('div');
      row.className = 'lanerow';

      const pattern = bar.lanes[lane.key] || REST.repeat(res);
      for (let slot = 0; slot < res; slot++) {
        row.appendChild(this._renderCell(bar, lane, slot, pattern[slot], slotsPerBeat));
      }
      grid.appendChild(row);
    }

    wrap.appendChild(grid);
    return wrap;
  }

  _renderCell(bar, lane, slot, char, slotsPerBeat) {
    const cell = document.createElement('button');
    cell.className = 'cell';
    cell.dataset.char = char;
    if (slot % slotsPerBeat === 0) cell.classList.add('beat');
    if (char !== REST) cell.classList.add('on');
    cell.textContent = char === REST ? '' : char;
    cell.title = `${lane.label} · slot ${slot + 1}`;

    const apply = (value) => {
      this._setCell(bar, lane.key, slot, value);
      cell.dataset.char = value;
      cell.textContent = value === REST ? '' : value;
      cell.classList.toggle('on', value !== REST);
    };

    cell.addEventListener('mousedown', (ev) => {
      ev.preventDefault();
      const cycle = cycleFor(lane.key);
      const current = cell.dataset.char;
      // Right-click clears; left-click advances through the cycle.
      const next = ev.button === 2
        ? REST
        : cycle[(cycle.indexOf(current) + 1) % cycle.length];
      apply(next);
      this._painting = next;
    });

    cell.addEventListener('mouseover', () => {
      if (this._painting !== null) apply(this._painting);
    });

    cell.addEventListener('contextmenu', (ev) => ev.preventDefault());
    return cell;
  }

  _setCell(bar, laneKey, slot, value) {
    const res = this.chart.res;
    let pattern = bar.lanes[laneKey] || REST.repeat(res);
    pattern = pattern.slice(0, slot) + value + pattern.slice(slot + 1);

    if (pattern === REST.repeat(res)) {
      delete bar.lanes[laneKey];   // keep empty lanes out of the file
    } else {
      bar.lanes[laneKey] = pattern;
    }
    this._changed();
  }

  _copyToNext(bar) {
    const idx = this.chart.bars.findIndex((b) => b.n === bar.n);
    const next = this.chart.bars[idx + 1];
    if (!next) return;
    next.lanes = JSON.parse(JSON.stringify(bar.lanes));
    this.render();
    this._changed();
  }

  _changed() {
    this.onChange(this.chart);
  }
}
