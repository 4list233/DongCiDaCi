/**
 * A timeline for the whole song: where you are, and how to get somewhere else.
 *
 * Reading a chart is not a linear pass. You play four bars, miss the fill, go
 * back four bars, try again. Without a way to say "there" that costs one click,
 * every retry means scrolling the score and hunting for the bar you wanted.
 *
 * Positions are held in bars rather than seconds, because a bar is the unit
 * you think in and because the chart's sync points already map bars to real
 * time -- including for a performance whose tempo moves.
 */

const MIN_TICK_SPACING_PX = 44;

export class Timeline {
  /**
   * @param {HTMLElement} host
   * @param {object} opts
   * @param {(bar:number) => void} opts.onSeek     jump to this bar
   * @param {(from:number, to:number|null) => void} opts.onLoop  set or clear a loop
   */
  constructor(host, { onSeek, onLoop } = {}) {
    this.host = host;
    this.onSeek = onSeek || (() => {});
    this.onLoop = onLoop || (() => {});

    this.chart = null;
    this.bar = 1;
    this.loop = null;         // {from, to} inclusive bar numbers
    this._dragFrom = null;

    this.host.classList.add('timeline');
    this.host.setAttribute('role', 'slider');
    this.host.setAttribute('aria-label', 'Position in the song');
    this.host.tabIndex = 0;

    this._build();
    this._bind();
  }

  _build() {
    this.host.replaceChildren();

    this.track = document.createElement('div');
    this.track.className = 'tl-track';

    this.loopBand = document.createElement('div');
    this.loopBand.className = 'tl-loop';
    this.loopBand.hidden = true;

    this.played = document.createElement('div');
    this.played.className = 'tl-played';

    this.ticks = document.createElement('div');
    this.ticks.className = 'tl-ticks';

    this.head = document.createElement('div');
    this.head.className = 'tl-head';

    this.readout = document.createElement('span');
    this.readout.className = 'tl-readout';

    this.track.append(this.loopBand, this.played, this.ticks, this.head);
    this.host.append(this.track, this.readout);
  }

  setChart(chart) {
    this.chart = chart;
    this.bar = 1;
    this.loop = null;
    this.loopBand.hidden = true;
    this.host.hidden = !chart;
    if (chart) {
      this.host.setAttribute('aria-valuemin', '1');
      this.host.setAttribute('aria-valuemax', String(this.totalBars));
      this._renderTicks();
      this.setBar(1);
    }
  }

  get totalBars() {
    return this.chart?.bars?.length || 0;
  }

  /** Move the playhead. Called from playback, so it must stay cheap. */
  setBar(bar) {
    if (!this.chart || !bar) return;
    this.bar = Math.min(Math.max(bar, 1), this.totalBars);

    const fraction = (this.bar - 1) / Math.max(this.totalBars - 1, 1);
    const percent = `${(fraction * 100).toFixed(2)}%`;
    this.head.style.left = percent;
    this.played.style.width = percent;

    this.host.setAttribute('aria-valuenow', String(this.bar));
    this.readout.textContent = this.loop
      ? `bar ${this.bar} / ${this.totalBars}  ·  loop ${this.loop.from}–${this.loop.to}`
      : `bar ${this.bar} / ${this.totalBars}`;
  }

  /**
   * Bar numbers along the track, thinned to whatever fits.
   *
   * A tick every bar is unreadable past about thirty bars and unusable on a
   * phone, so the interval grows until the labels have room.
   */
  _renderTicks() {
    this.ticks.replaceChildren();
    const total = this.totalBars;
    if (!total) return;

    const width = this.host.getBoundingClientRect().width || 800;
    const step = niceStep(total, width);

    for (let bar = 1; bar <= total; bar += step) {
      const tick = document.createElement('span');
      tick.className = 'tl-tick';
      tick.style.left = `${((bar - 1) / Math.max(total - 1, 1) * 100).toFixed(2)}%`;
      tick.textContent = String(bar);
      this.ticks.appendChild(tick);
    }

    // Sections are the landmarks people actually navigate by.
    for (const bar of this.chart.bars) {
      if (!bar.section) continue;
      const mark = document.createElement('span');
      mark.className = 'tl-section';
      mark.style.left = `${((bar.n - 1) / Math.max(total - 1, 1) * 100).toFixed(2)}%`;
      mark.title = `${bar.section} (bar ${bar.n})`;
      mark.textContent = bar.section;
      this.ticks.appendChild(mark);
    }
  }

  relayout() {
    if (this.chart) this._renderTicks();
  }

  _barAtClientX(clientX) {
    const box = this.track.getBoundingClientRect();
    const fraction = Math.min(Math.max((clientX - box.left) / box.width, 0), 1);
    return Math.round(fraction * (this.totalBars - 1)) + 1;
  }

  _bind() {
    // Click seeks. Drag across selects a range to loop, because "play these
    // four bars again" is the single most common thing you want from a chart.
    this.track.addEventListener('pointerdown', (event) => {
      if (!this.chart) return;
      this._dragFrom = this._barAtClientX(event.clientX);
      this.track.setPointerCapture(event.pointerId);
      event.preventDefault();
    });

    this.track.addEventListener('pointermove', (event) => {
      if (this._dragFrom == null) return;
      const to = this._barAtClientX(event.clientX);
      if (Math.abs(to - this._dragFrom) >= 1) this._showLoop(this._dragFrom, to);
    });

    const finish = (event) => {
      if (this._dragFrom == null) return;
      const to = this._barAtClientX(event.clientX);
      const from = this._dragFrom;
      this._dragFrom = null;

      if (Math.abs(to - from) < 1) {
        this.clearLoop();
        this.setBar(from);
        this.onSeek(from);
      } else {
        const [lo, hi] = from < to ? [from, to] : [to, from];
        this._showLoop(lo, hi);
        this.loop = { from: lo, to: hi };
        this.setBar(lo);
        this.onLoop(lo, hi);
      }
    };

    this.track.addEventListener('pointerup', finish);
    this.track.addEventListener('pointercancel', () => { this._dragFrom = null; });

    // Double-click clears the loop, so escaping it does not need a menu.
    this.track.addEventListener('dblclick', () => {
      this.clearLoop();
      this.onLoop(null, null);
    });

    this.host.addEventListener('keydown', (event) => {
      if (!this.chart) return;
      const step = event.shiftKey ? 4 : 1;
      let target = null;
      if (event.key === 'ArrowLeft') target = this.bar - step;
      else if (event.key === 'ArrowRight') target = this.bar + step;
      else if (event.key === 'Home') target = 1;
      else if (event.key === 'End') target = this.totalBars;
      if (target == null) return;

      event.preventDefault();
      const bar = Math.min(Math.max(target, 1), this.totalBars);
      this.setBar(bar);
      this.onSeek(bar);
    });
  }

  _showLoop(from, to) {
    const [lo, hi] = from < to ? [from, to] : [to, from];
    const total = Math.max(this.totalBars - 1, 1);
    this.loopBand.hidden = false;
    this.loopBand.style.left = `${((lo - 1) / total * 100).toFixed(2)}%`;
    this.loopBand.style.width = `${((hi - lo) / total * 100).toFixed(2)}%`;
  }

  clearLoop() {
    this.loop = null;
    this.loopBand.hidden = true;
    this.setBar(this.bar);
  }
}

/** Tick interval that keeps labels from colliding at this width. */
function niceStep(total, widthPx) {
  const maxTicks = Math.max(Math.floor(widthPx / MIN_TICK_SPACING_PX), 2);
  const raw = total / maxTicks;
  for (const step of [1, 2, 4, 8, 16, 32, 64]) {
    if (step >= raw) return step;
  }
  return 128;
}
