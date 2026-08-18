/**
 * Layout that adapts rather than assuming a screen.
 *
 * Two things were fixed that should never have been: the sidebar always took
 * 19rem whether or not you were using it, and the notation got 42% of the height
 * regardless of whether you were reading the part or fixing it. Both are now
 * yours to set, and both are remembered, because a layout you have to
 * re-establish every visit is not adjustable in any useful sense.
 */

const SPLIT_KEY = 'dcdc.layout.split';
const SIDEBAR_KEY = 'dcdc.layout.sidebar';

// Leave enough of each pane to still be usable, and to leave the splitter
// somewhere it can be grabbed to undo a drag that went too far.
const MIN_SHARE = 0.15;
const MAX_SHARE = 0.85;

// Below this the sidebar overlays rather than sits alongside, matching the CSS
// breakpoint. Kept in one place so the two cannot disagree.
const NARROW_PX = 900;

export function initLayout({ onResize } = {}) {
  const splitter = document.querySelector('#splitter');
  const main = document.querySelector('.main');
  const toggle = document.querySelector('#sidebartoggle');

  restoreSplit(main);
  restoreSidebar(toggle);

  bindSplitter(splitter, main, onResize);
  bindSidebar(toggle);
}

// --- the split --------------------------------------------------------------

function restoreSplit(main) {
  const stored = Number(localStorage.getItem(SPLIT_KEY));
  if (stored >= MIN_SHARE && stored <= MAX_SHARE) setSplit(main, stored);
}

function setSplit(main, share) {
  main.style.setProperty('--split', `${(share * 100).toFixed(1)}%`);
}

function bindSplitter(splitter, main, onResize) {
  if (!splitter || !main) return;

  let dragging = false;

  const apply = (clientY) => {
    const box = main.getBoundingClientRect();
    const notation = document.querySelector('.notation-wrap');
    const top = notation ? notation.getBoundingClientRect().top : box.top;

    // The share is a flex-basis percentage, and flex-basis resolves against the
    // *container's* height -- not against the space left below the toolbars.
    // Dividing by the remaining space instead made every drag land short of
    // where it was dropped, by however tall the rows above happened to be.
    const height = box.height;
    if (height <= 0) return;

    const share = Math.min(Math.max((clientY - top) / height, MIN_SHARE), MAX_SHARE);
    setSplit(main, share);
    localStorage.setItem(SPLIT_KEY, String(share));
    onResize?.();
  };

  const stop = () => {
    if (!dragging) return;
    dragging = false;
    document.body.classList.remove('dragging');
  };

  splitter.addEventListener('pointerdown', (event) => {
    dragging = true;
    document.body.classList.add('dragging');
    // Keep receiving moves even when the pointer outruns the 7px handle.
    splitter.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  splitter.addEventListener('pointermove', (event) => {
    if (dragging) apply(event.clientY);
  });

  splitter.addEventListener('pointerup', stop);
  splitter.addEventListener('pointercancel', stop);

  // Keyboard: a splitter that only responds to a precise drag excludes anyone
  // not using a mouse, and is fiddly even for those who are.
  splitter.addEventListener('keydown', (event) => {
    const step = event.shiftKey ? 0.1 : 0.02;
    const current = currentShare(main);
    if (event.key === 'ArrowUp') {
      setStoredSplit(main, current - step, onResize);
    } else if (event.key === 'ArrowDown') {
      setStoredSplit(main, current + step, onResize);
    } else {
      return;
    }
    event.preventDefault();
  });

  // Double-click restores the default rather than making you hunt for it.
  splitter.addEventListener('dblclick', () => setStoredSplit(main, 0.42, onResize));
}

function currentShare(main) {
  const value = getComputedStyle(main).getPropertyValue('--split').trim();
  const parsed = parseFloat(value);
  return Number.isFinite(parsed) ? parsed / 100 : 0.42;
}

function setStoredSplit(main, share, onResize) {
  const clamped = Math.min(Math.max(share, MIN_SHARE), MAX_SHARE);
  setSplit(main, clamped);
  localStorage.setItem(SPLIT_KEY, String(clamped));
  onResize?.();
}

// --- the sidebar ------------------------------------------------------------

function restoreSidebar(toggle) {
  // Open on a wide screen, closed on a narrow one, unless told otherwise.
  const stored = localStorage.getItem(SIDEBAR_KEY);
  const open = stored === null ? window.innerWidth > NARROW_PX : stored === 'open';
  applySidebar(toggle, open);
}

function applySidebar(toggle, open) {
  document.body.classList.toggle('sidebar-open', open);
  toggle?.setAttribute('aria-expanded', String(open));
}

function bindSidebar(toggle) {
  if (!toggle) return;

  toggle.addEventListener('click', () => {
    const open = !document.body.classList.contains('sidebar-open');
    applySidebar(toggle, open);
    localStorage.setItem(SIDEBAR_KEY, open ? 'open' : 'closed');
  });

  // On a narrow screen the sidebar covers the score, so picking a song should
  // hand the screen back rather than leaving you to dismiss it.
  document.querySelector('#library')?.addEventListener('click', () => {
    if (window.innerWidth <= NARROW_PX) {
      applySidebar(toggle, false);
      localStorage.setItem(SIDEBAR_KEY, 'closed');
    }
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && document.body.classList.contains('sidebar-open')
        && window.innerWidth <= NARROW_PX) {
      applySidebar(toggle, false);
    }
  });
}
