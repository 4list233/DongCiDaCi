/** Thin wrapper over the backend. Local-only, so no auth and no retries. */

const BASE = import.meta.env.DEV ? 'http://127.0.0.1:8000' : '';

async function request(path, options = {}) {
  const res = await fetch(BASE + path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

const json = (body) => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request('/api/health'),
  vocabulary: () => request('/api/vocabulary'),

  listSongs: () => request('/api/songs'),
  getSong: (slug) => request(`/api/songs/${slug}`),
  createSong: (body) => request('/api/songs', json(body)),
  deleteSong: (slug) => request(`/api/songs/${slug}`, { method: 'DELETE' }),

  uploadAudio: (slug, file) => {
    const form = new FormData();
    form.append('file', file);
    return request(`/api/songs/${slug}/audio`, { method: 'POST', body: form });
  },
  audioUrl: (slug) => `${BASE}/api/songs/${slug}/audio`,

  transcribe: (slug, options = {}) => request(`/api/songs/${slug}/transcribe`, json(options)),

  createFromUrl: (body) => request('/api/songs/from-url', json(body)),
  fetchAudio: (slug, body) => request(`/api/songs/${slug}/fetch`, json(body)),

  getChart: (slug) => request(`/api/songs/${slug}/chart`),
  getRawChart: (slug) => request(`/api/songs/${slug}/raw`),
  putChart: (slug, chart) => request(`/api/songs/${slug}/chart`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(chart),
  }),
};

/** Poll a running job until it finishes. */
export function watchJob(slug, onUpdate, interval = 1200) {
  let stopped = false;
  (async function poll() {
    while (!stopped) {
      try {
        const song = await api.getSong(slug);
        onUpdate(song);
        if (song.status === 'ready' || song.status === 'failed') return;
      } catch (err) {
        onUpdate({ status: 'failed', error: err.message });
        return;
      }
      await new Promise((r) => setTimeout(r, interval));
    }
  })();
  return () => { stopped = true; };
}
