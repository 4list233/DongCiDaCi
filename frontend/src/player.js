/**
 * Notation rendering and playback.
 *
 * Two sound sources, and choosing between them is the point of the app:
 *
 *   backing track  the original recording, with the cursor following it via
 *                  sync points derived from the beat tracker. This is how you
 *                  check a chart against what was actually played.
 *   synth          alphaSynth playing your edited chart. This is how you hear
 *                  the version you decided on.
 *
 * The backing-track wiring follows alphaTab's actual model rather than a
 * plausible-looking API: sync points are `Automation` objects of type
 * `SyncPoint` attached to the score's master bars, and `updateSyncPoints()`
 * takes no arguments -- it reads them back off the score.
 */

import * as alphaTab from '@coderline/alphatab';
import { chartToAlphaTex } from './alphatex.js';

export class Player {
  constructor(host, { onBarChange, onReady } = {}) {
    this.host = host;
    this.onBarChange = onBarChange || (() => {});
    this.onReady = onReady || (() => {});
    this.chart = null;
    this.articulations = undefined;
    this.audioBytes = null;
    this._lastBar = null;

    this.api = new alphaTab.AlphaTabApi(host, {
      core: { fontDirectory: '/font/' },
      display: { scale: 0.9, layoutMode: alphaTab.LayoutMode.Page },
      player: {
        playerMode: alphaTab.PlayerMode.EnabledAutomatic,
        enableCursor: true,
        enableUserInteraction: true,
        soundFont: '/soundfont/sonivox.sf2',
        scrollElement: host.parentElement || host,
      },
    });

    this.api.playerReady.on(() => this.onReady());

    // Sync points have to be re-attached every time a score is (re)loaded,
    // because loading builds a fresh model from the AlphaTex.
    this.api.scoreLoaded.on(() => this._attachSyncPoints());

    // alphaTab reports position continuously; only bar changes are interesting.
    this.api.playerPositionChanged.on((args) => {
      const bar = this._barAt(args.currentTime / 1000);
      if (bar !== this._lastBar) {
        this._lastBar = bar;
        this.onBarChange(bar);
      }
    });
  }

  setVocabulary(lanes) {
    this.articulations = Object.fromEntries(lanes.map((l) => [l.key, l.alphatex]));
  }

  /** Re-render notation from the chart. Called on every edit. */
  setChart(chart) {
    this.chart = chart;
    this.api.tex(chartToAlphaTex(chart, this.articulations));
  }

  /**
   * Attach the original recording as a backing track.
   *
   * alphaTab wants the raw bytes on `score.backingTrack`, not a URL, so the
   * file is fetched once and kept for re-application after each re-render.
   */
  async setBackingTrack(url) {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`audio fetch failed: ${response.status}`);
      this.audioBytes = new Uint8Array(await response.arrayBuffer());
      this._applyBackingTrack();
    } catch (err) {
      console.warn('backing track unavailable, staying on the synth:', err);
      this.audioBytes = null;
    }
  }

  _applyBackingTrack() {
    const score = this.api.score;
    if (!score || !this.audioBytes) return;

    const backingTrack = new alphaTab.model.BackingTrack();
    backingTrack.rawAudioFile = this.audioBytes;
    score.backingTrack = backingTrack;

    this._attachSyncPoints();

    this.api.settings.player.playerMode = alphaTab.PlayerMode.EnabledBackingTrack;
    this.api.updateSettings();
    this.api.renderScore(score);
  }

  /**
   * Push the chart's sync points onto the score's master bars.
   *
   * One point per bar, at ratioPosition 0 (the downbeat). That is enough for
   * alphaTab to interpolate the cursor across a performance that drifts, which
   * every human-played record does.
   */
  _attachSyncPoints() {
    const score = this.api.score;
    if (!score || !this.chart?.sync?.length) return;

    for (const point of this.chart.sync) {
      const masterBar = score.masterBars[point.bar - 1];
      if (!masterBar) continue;

      const automation = new alphaTab.model.Automation();
      automation.type = alphaTab.model.AutomationType.SyncPoint;
      automation.ratioPosition = 0;

      const data = new alphaTab.model.SyncPointData();
      data.barOccurence = 0;
      data.millisecondOffset = Math.round(point.time * 1000);
      automation.syncPointValue = data;

      masterBar.syncPoints = undefined;   // replace rather than accumulate
      masterBar.addSyncPoint(automation);
    }

    this.api.updateSyncPoints();
  }

  /** Which bar is sounding at a given wall-clock second in the recording. */
  _barAt(seconds) {
    const sync = this.chart?.sync;
    if (!sync?.length) return null;
    let bar = sync[0].bar;
    for (const point of sync) {
      if (point.time <= seconds) bar = point.bar;
      else break;
    }
    return bar;
  }

  playFromBar(barNumber) {
    const masterBar = this.api.score?.masterBars?.[barNumber - 1];
    if (masterBar) this.api.tickPosition = masterBar.start;
    this.api.play();
  }

  /** Loop a bar range -- the thing you actually do when learning a part. */
  loopBars(from, to) {
    const bars = this.api.score?.masterBars || [];
    const start = bars[from - 1];
    const end = bars[to];      // exclusive: the start of the bar after the range
    if (!start) return;

    this.api.playbackRange = {
      startTick: start.start,
      endTick: end ? end.start : this.api.endTick,
    };
    this.api.isLooping = true;
    this.api.play();
  }

  clearLoop() {
    this.api.isLooping = false;
    this.api.playbackRange = null;
  }

  /** Switch back to hearing your own chart instead of the record. */
  useSynth() {
    this.api.settings.player.playerMode = alphaTab.PlayerMode.EnabledSynthesizer;
    this.api.updateSettings();
  }

  useBackingTrack() {
    if (this.audioBytes) this._applyBackingTrack();
  }

  setSpeed(factor) {
    this.api.playbackSpeed = factor;
  }

  playPause() { this.api.playPause(); }
  stop() { this.api.stop(); }
}
