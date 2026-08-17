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

// How far the stripped recording may drift from the score before it is
// pulled back. Below roughly this, a listener hears one performance.
const STEM_DRIFT_TOLERANCE_S = 0.05;

export class Player {
  constructor(host, { onBarChange, onReady, onStemError } = {}) {
    this.host = host;
    this.onBarChange = onBarChange || (() => {});
    this.onReady = onReady || (() => {});
    this.onStemError = onStemError || (() => {});
    this.mode = 'original';
    this.originalUrl = null;
    this.stemUrl = null;
    this.stem = null;
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
      // In 'both' mode the synth is the clock, so this is also where the
      // stripped recording gets pulled back into line.
      if (this.stem) this._followStem(args.currentTick);

      const bar = this.mode === 'both'
        ? this._barAtTick(args.currentTick)
        : this._barAt(args.currentTime / 1000);
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

  /** Which bar a score tick falls in. Used when the synth drives playback. */
  _barAtTick(tick) {
    const bars = this.api.score?.masterBars;
    if (!bars?.length) return null;
    let index = 0;
    for (let i = 0; i < bars.length; i += 1) {
      if (bars[i].start <= tick) index = i;
      else break;
    }
    return index + 1;
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

  /**
   * Choose what you hear.
   *
   *   original   the record as released
   *   nodrums    the record with the drums removed -- what you play along to
   *   chart      only the transcription, on the synth
   *   both       the stripped record with the transcribed drums played over it,
   *              which is the only way to judge whether the chart is *right*
   *              rather than merely plausible
   *
   * `both` cannot use alphaTab's backing-track mode, because that mode plays
   * audio instead of synthesizing rather than as well. So the synth is the clock
   * and the stripped recording is slaved to it.
   */
  async setMode(mode) {
    this.mode = mode;
    this._detachStem();

    if (mode === 'chart') {
      this.api.settings.player.playerMode = alphaTab.PlayerMode.EnabledSynthesizer;
      this.api.updateSettings();
      return;
    }

    if (mode === 'both') {
      this.api.settings.player.playerMode = alphaTab.PlayerMode.EnabledSynthesizer;
      this.api.updateSettings();
      await this._attachStem();
      return;
    }

    // original / nodrums: alphaTab plays the audio and follows it via sync
    // points, which is more accurate than anything done by hand.
    const url = mode === 'nodrums' ? this.stemUrl : this.originalUrl;
    if (!url) return;
    await this.setBackingTrack(url);
  }

  /**
   * Play the drums-removed recording underneath the synthesized chart.
   *
   * The two are kept together by correcting the audio toward where the score
   * says it should be, rather than starting both and hoping. A performance
   * drifts against a fixed tempo, so without correction they separate audibly
   * within a chorus.
   */
  async _attachStem() {
    if (!this.stemUrl) return;

    const audio = new Audio(this.stemUrl);
    audio.preload = 'auto';
    audio.playbackRate = this.api.playbackSpeed || 1;

    // Report a missing stem rather than failing silently to no sound at all.
    audio.addEventListener('error', () => {
      this.onStemError?.('no drums-removed stem for this song — re-transcribe to produce one');
      this._detachStem();
    });

    this.stem = audio;
  }

  _detachStem() {
    if (!this.stem) return;
    this.stem.pause();
    this.stem = null;
  }

  /** Nudge the stem back toward the score position when it has drifted. */
  _followStem(currentTick) {
    if (!this.stem || this.stem.readyState < 2) return;

    const target = this._recordingTimeAtTick(currentTick);
    if (target == null) return;

    const drift = Math.abs(this.stem.currentTime - target);
    // Correct only past what a listener would notice. Constant nudging is
    // audible as a stutter and is worse than the drift it fixes.
    if (drift > STEM_DRIFT_TOLERANCE_S) this.stem.currentTime = target;
  }

  /** Where in the recording a score tick falls, via the chart's sync points. */
  _recordingTimeAtTick(tick) {
    const sync = this.chart?.sync;
    const bars = this.api.score?.masterBars;
    if (!sync?.length || !bars?.length) return null;

    let index = 0;
    for (let i = 0; i < bars.length; i += 1) {
      if (bars[i].start <= tick) index = i;
      else break;
    }

    const point = sync[index] || sync[sync.length - 1];
    const next = sync[index + 1];
    const barStart = bars[index].start;
    const barTicks = (bars[index + 1]?.start ?? barStart + 3840) - barStart;
    if (!barTicks) return point.time;

    const within = Math.min(Math.max((tick - barStart) / barTicks, 0), 1);
    const span = next ? next.time - point.time : 0;
    return point.time + within * span;
  }

  /** Switch back to hearing your own chart instead of the record. */
  useSynth() { return this.setMode('chart'); }

  useBackingTrack() { return this.setMode('original'); }

  setSpeed(factor) {
    this.api.playbackSpeed = factor;
    if (this.stem) this.stem.playbackRate = factor;
  }

  playPause() {
    this.api.playPause();
    if (this.stem) {
      if (this.api.playerState === alphaTab.synth.PlayerState.Playing) {
        this.stem.play().catch(() => {});
      } else {
        this.stem.pause();
      }
    }
  }

  stop() {
    this.api.stop();
    if (this.stem) {
      this.stem.pause();
      this.stem.currentTime = 0;
    }
  }
}
