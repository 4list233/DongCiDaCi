/** Wires the library, pipeline status, notation, and grid editor together. */

import { api, watchJob } from './api.js';
import { GridEditor } from './editor.js';
import { Player } from './player.js';
import { initLayout } from './layout.js';
import './style.css';

const $ = (sel) => document.querySelector(sel);

const state = {
  slug: null,
  chart: null,
  lanes: [],
  dirty: false,
  saveTimer: null,
  // How this chart was read out of the audio, mirrored by the reading controls.
  reading: { sensitivity: 0.5, offset_beats: 0, res: '' },
};

let editor = null;
let player = null;

// --- boot -------------------------------------------------------------------

async function boot() {
  const [health, vocab] = await Promise.all([api.health(), api.vocabulary()]);
  state.lanes = vocab.lanes;
  renderCapabilities(health);

  editor = new GridEditor($('#grid'), {
    lanes: state.lanes,
    onChange: onChartEdited,
    onSeek: (bar) => player?.playFromBar(bar),
  });

  player = new Player($('#notation'), {
    onBarChange: (bar) => editor.setActiveBar(bar),
    onStemError: (message) => setStatus(message),
  });
  player.setVocabulary(state.lanes);

  bindControls();
  // alphaTab lays out to the width it is given, so a resize has to tell it.
  initLayout({ onResize: () => player?.api?.render?.() });
  await refreshLibrary();

  const slug = new URLSearchParams(location.search).get('song');
  if (slug) await openSong(slug);
}

/**
 * Say out loud which backends are live. Running on the spectral fallback and
 * believing you got a real transcription is the worst possible failure mode,
 * so it gets a permanent banner rather than a console log.
 */
function renderCapabilities(health) {
  const el = $('#capabilities');
  const bits = [
    `device ${health.device}`,
    health.demucs ? 'demucs ok' : 'demucs MISSING',
    health.adtof ? 'adtof ok' : 'adtof missing — spectral fallback',
  ];
  if (!health.ytdlp) bits.push('yt-dlp missing — links disabled');
  el.textContent = bits.join(' · ');
  el.classList.toggle('warn', !health.adtof || !health.demucs || !health.ytdlp);

  // Do not offer a control that cannot work.
  if (!health.ytdlp) {
    for (const id of ['#urlinput', '#addbtn', '#fetchurl', '#fetchbtn']) {
      const node = $(id);
      if (node) node.disabled = true;
    }
    $('#urlinput').placeholder = "yt-dlp not installed — pip install -e '.[fetch]'";
    $('#addbtn').textContent = 'Add song';
  }
}

// --- library ----------------------------------------------------------------

async function refreshLibrary() {
  const songs = await api.listSongs();
  const list = $('#library');
  list.replaceChildren();

  for (const song of songs) {
    const item = document.createElement('button');
    item.className = 'songitem';
    if (song.slug === state.slug) item.classList.add('active');

    const title = document.createElement('span');
    title.className = 'songtitle';
    title.textContent = song.title;

    const meta = document.createElement('span');
    meta.className = 'songmeta';
    meta.textContent = song.artist || '—';

    const status = document.createElement('span');
    status.className = `status ${song.status}`;
    status.textContent = song.status === 'running'
      ? `${Math.round(song.progress * 100)}%`
      : song.status;

    item.append(title, meta, status);
    item.addEventListener('click', () => openSong(song.slug));
    list.appendChild(item);
  }
}

async function openSong(slug) {
  state.slug = slug;
  // Asked once per song, not once per drag of the slider.
  state.rereadConfirmed = false;
  history.replaceState(null, '', `?song=${slug}`);

  const song = await api.getSong(slug);
  $('#songtitle').textContent = song.title;
  $('#songartist').textContent = song.artist || '';
  await refreshLibrary();

  if (song.status === 'running' || song.status === 'queued') {
    trackJob(slug);
  }

  renderReport(song);

  // A song with no audio can do nothing at all, so say so with a control
  // attached rather than an instruction the UI does not let you follow.
  $('#needaudio').hidden = song.has_audio;
  $('#run').disabled = !song.has_audio;
  $('#run').title = song.has_audio
    ? 'Run the transcription pipeline'
    : 'Upload an audio file first';
  $('#deletesong').hidden = false;

  if (song.has_chart) {
    await loadChart(slug);
    if (song.has_audio) {
      // Both sources up front: switching what you hear should not wait on a
      // fetch, and the drums-removed stem is what the play-along modes need.
      player.originalUrl = api.audioUrl(slug);
      player.stemUrl = api.stemUrl(slug, 'no_drums');
      await player.setMode($('#source').value || 'original');
    }
  } else {
    state.chart = null;
    editor.setChart(null);
    $('#reading').hidden = true;
    setStatus(song.has_audio
      ? 'Audio ready. Hit Transcribe for a first draft.'
      : 'No audio yet — this song cannot be transcribed until you add a file.');
  }
}

async function loadChart(slug, { keepReading = false } = {}) {
  state.chart = await api.getChart(slug);
  editor.setChart(state.chart);
  player.setChart(state.chart);
  setStatus(`${state.chart.bars.length} bars · res ${state.chart.res} · ${state.chart.tempo} bpm`);

  // Reflect whatever this chart was last built with, so the controls describe
  // the chart on screen rather than resetting to defaults every time. After a
  // reread the controls are already correct and re-deriving them would discard
  // the grid choice, which is not recorded on the chart.
  if (!keepReading) {
    const source = state.chart.source || {};
    state.reading = {
      sensitivity: source.sensitivity ?? 0.5,
      offset_beats: source.offset_beats ?? 0,
      res: '',
    };
  }
  syncReadingControls();
}

// --- rereading --------------------------------------------------------------

/**
 * Rebuild the chart from cached analysis.
 *
 * These are the corrections that are not editing: how much of what was detected
 * counts as played, and where bar one is. They rebuild rather than modify, so
 * they cannot be applied to a chart you have already edited by hand without
 * discarding that work -- which is why this asks first.
 */
function syncReadingControls() {
  const reading = state.reading || { sensitivity: 0.5, offset_beats: 0, res: '' };
  $('#reading').hidden = !state.chart;
  $('#sensitivity').value = reading.sensitivity;
  $('#senslabel').textContent = `${Math.round(reading.sensitivity * 100)}%`;
  $('#gridoffset').textContent = reading.offset_beats > 0
    ? `+${reading.offset_beats}` : `${reading.offset_beats}`;
  $('#res').value = reading.res || '';
}

async function reread(changes) {
  if (!state.slug || !state.chart) return;

  if (state.dirty && !state.rereadConfirmed) {
    const ok = confirm(
      'This rebuilds the chart from the original audio analysis and will discard '
      + 'your edits to it. Continue?'
    );
    if (!ok) {
      syncReadingControls();     // put the control back where it was
      return;
    }
    state.rereadConfirmed = true;
  }

  state.reading = { ...state.reading, ...changes };
  syncReadingControls();

  // Coalesce a dragged slider into one request per pause.
  clearTimeout(state.rereadTimer);
  state.rereadTimer = setTimeout(async () => {
    const hint = $('#readingstatus');
    hint.textContent = 'rereading…';
    try {
      const result = await api.requantize(state.slug, {
        sensitivity: state.reading.sensitivity,
        offset_beats: state.reading.offset_beats,
        res: state.reading.res ? Number(state.reading.res) : null,
      });
      state.dirty = false;
      await loadChart(state.slug, { keepReading: true });
      renderReport({ report: result });
      hint.textContent = '';
    } catch (err) {
      hint.textContent = err.message;
    }
  }, 180);
}

// --- editing ----------------------------------------------------------------

/** Autosave, debounced. There is no explicit save button on purpose -- the
 *  file on disk should always match what you see. */
function onChartEdited(chart) {
  state.dirty = true;
  player.setChart(chart);
  setStatus('edited…');

  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(async () => {
    try {
      await api.putChart(state.slug, chart);
      state.dirty = false;
      setStatus('saved');
    } catch (err) {
      setStatus(`save failed: ${err.message}`, true);
    }
  }, 700);
}

// --- pipeline ---------------------------------------------------------------

function trackJob(slug) {
  setStatus('transcribing…');
  watchJob(slug, async (song) => {
    if (song.status === 'running') {
      setStatus(`${song.stage} — ${Math.round(song.progress * 100)}%`);
    } else if (song.status === 'ready') {
      setStatus('transcription complete');
      renderReport(song);
      await loadChart(slug);
      const full = await api.getSong(slug);
      if (full.has_audio) player.setBackingTrack(api.audioUrl(slug));
    } else if (song.status === 'failed') {
      setStatus(`failed: ${song.error}`, true);
    }
    refreshLibrary();
  });
}

/** Surface the quantizer's own warnings. It knows when it did a bad job. */
function renderReport(song) {
  const el = $('#report');
  const report = song.report || {};
  el.replaceChildren();
  if (!report.warnings?.length) return;

  const heading = document.createElement('strong');
  heading.textContent = `res ${report.res} · grid fit ${(report.fit_error * 100).toFixed(0)}%`
    + (report.flams ? ` · ${report.flams} flams` : '')
    + (report.dropped ? ` · ${report.dropped} collisions` : '');
  el.appendChild(heading);

  const ul = document.createElement('ul');
  for (const warning of report.warnings) {
    const li = document.createElement('li');
    li.textContent = warning;
    ul.appendChild(li);
  }
  el.appendChild(ul);
}

// --- controls ---------------------------------------------------------------

function bindControls() {
  // Show which file was picked. Without this there is no confirmation that the
  // browser took it, which is how you end up submitting an empty form twice.
  $('#filepick').querySelector('input').addEventListener('change', (ev) => {
    const file = ev.target.files?.[0];
    $('#filename').textContent = file ? file.name : 'Choose audio file';
    $('#filepick').classList.toggle('chosen', Boolean(file));
  });

  $('#whynot').addEventListener('click', (ev) => {
    ev.preventDefault();
    $('#needaudio').hidden = false;
    $('#needaudio').scrollIntoView({ block: 'nearest' });
  });

  // A link fetches and transcribes in one go; a file uploads and waits. Both
  // come through the same form, because "add a song" is one intention.
  $('#newsong').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const form = new FormData(ev.target);
    const url = String(form.get('url') || '').trim();
    const file = form.get('audio');
    const title = String(form.get('title') || '').trim();
    const artist = String(form.get('artist') || '').trim();

    if (!url && !(file && file.size)) {
      setStatus('Paste a link or choose an audio file.', true);
      return;
    }

    const submit = $('#addbtn');
    const original = submit.textContent;
    submit.disabled = true;

    try {
      if (url) {
        submit.textContent = 'Fetching…';
        const { slug } = await api.createFromUrl({ url, title, artist, transcribe: true });
        resetForm(ev.target);
        await refreshLibrary();
        await openSong(slug);
        trackJob(slug);
      } else {
        submit.textContent = 'Uploading…';
        const song = await api.createSong({ title: title || file.name, artist });
        await api.uploadAudio(song.slug, file);
        resetForm(ev.target);
        await refreshLibrary();
        await openSong(song.slug);
      }
    } catch (err) {
      setStatus(err.message, true);
    } finally {
      submit.disabled = false;
      submit.textContent = original;
    }
  });

  // Fetch audio for a song that already exists.
  $('#fetchbtn').addEventListener('click', async () => {
    const url = $('#fetchurl').value.trim();
    if (!url || !state.slug) {
      setStatus('Paste a link first.', true);
      return;
    }
    $('#fetchbtn').disabled = true;
    try {
      await api.fetchAudio(state.slug, { url, transcribe: true });
      $('#fetchurl').value = '';
      trackJob(state.slug);
    } catch (err) {
      setStatus(err.message, true);
    } finally {
      $('#fetchbtn').disabled = false;
    }
  });

  // Attach audio to a song that was created without it.
  $('#addaudio').addEventListener('change', async (ev) => {
    const file = ev.target.files?.[0];
    if (!file || !state.slug) return;
    $('#addaudioname').textContent = `Uploading ${file.name}…`;
    try {
      await api.uploadAudio(state.slug, file);
      await openSong(state.slug);
      setStatus('Audio added. Hit Transcribe for a first draft.');
    } catch (err) {
      setStatus(`Upload failed: ${err.message}`, true);
    } finally {
      $('#addaudioname').textContent = 'Choose audio file';
      ev.target.value = '';
    }
  });

  $('#deletesong').addEventListener('click', async () => {
    if (!state.slug) return;
    const song = await api.getSong(state.slug);
    if (!confirm(`Delete "${song.title}" and its chart? This cannot be undone.`)) return;
    try {
      await api.deleteSong(state.slug);
      state.slug = null;
      state.chart = null;
      editor.setChart(null);
      history.replaceState(null, '', location.pathname);
      $('#songtitle').textContent = 'No song selected';
      $('#songartist').textContent = '';
      $('#deletesong').hidden = true;
      $('#needaudio').hidden = true;
      setStatus('Deleted.');
      await refreshLibrary();
    } catch (err) {
      setStatus(`Could not delete: ${err.message}`, true);
    }
  });

  $('#run').addEventListener('click', async () => {
    if (!state.slug) return;
    try {
      await api.transcribe(state.slug);
      trackJob(state.slug);
    } catch (err) {
      setStatus(err.message, true);
    }
  });

  $('#playpause').addEventListener('click', () => player.playPause());
  $('#stop').addEventListener('click', () => player.stop());

  $('#source').addEventListener('change', (ev) => player.setMode(ev.target.value));

  $('#sensitivity').addEventListener('input', (ev) => {
    const value = Number(ev.target.value);
    // Update the label immediately; the rebuild itself is debounced.
    $('#senslabel').textContent = `${Math.round(value * 100)}%`;
    reread({ sensitivity: value });
  });

  $('#gridback').addEventListener('click', () => {
    reread({ offset_beats: (state.reading?.offset_beats ?? 0) - 1 });
  });
  $('#gridfwd').addEventListener('click', () => {
    reread({ offset_beats: (state.reading?.offset_beats ?? 0) + 1 });
  });

  $('#res').addEventListener('change', (ev) => reread({ res: ev.target.value }));

  $('#speed').addEventListener('input', (ev) => {
    const factor = Number(ev.target.value);
    player.setSpeed(factor);
    $('#speedlabel').textContent = `${Math.round(factor * 100)}%`;
  });

  // Space toggles playback unless you are typing an annotation.
  document.addEventListener('keydown', (ev) => {
    if (ev.code === 'Space' && ev.target.tagName !== 'INPUT' && ev.target.tagName !== 'TEXTAREA') {
      ev.preventDefault();
      player.playPause();
    }
  });

  window.addEventListener('beforeunload', (ev) => {
    if (state.dirty) ev.preventDefault();
  });
}

function resetForm(form) {
  form.reset();
  $('#filename').textContent = 'Choose audio file';
  $('#filepick').classList.remove('chosen');
}

function setStatus(text, isError = false) {
  const el = $('#status');
  el.textContent = text;
  el.classList.toggle('error', isError);
}

boot().catch((err) => {
  document.body.insertAdjacentHTML(
    'afterbegin',
    `<div class="fatal">Could not reach the backend: ${err.message}<br>
     <code>uv run dcdc serve</code></div>`,
  );
});
