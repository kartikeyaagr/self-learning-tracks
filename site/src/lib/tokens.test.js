// Text contrast of the Unroll tokens (WCAG AA: 4.5:1 for body text), dark desktop and light mobile palettes.
// --faint is for glyphs only, so it isn't checked. npm test.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const css = fs.readFileSync(new URL('../styles/unroll/tokens.css', import.meta.url), 'utf8');
const block = sel => Object.fromEntries([...css.slice(css.indexOf(sel)).split('}')[0].matchAll(/--([\w-]+):\s*(#[0-9A-Fa-f]{6})/g)].map(m => [m[1], m[2]]));
const dark = block(':root,'), mobile = {...dark, ...block('[data-surface="mobile"]')};
mobile['accent-text'] = mobile['accent-edge'];   // styles/unroll.css: accent as text on phones
dark['accent-text'] = dark.accent;
const lum = h => { const [r, g, b] = [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16) / 255).map(c => c <= .03928 ? c / 12.92 : ((c + .055) / 1.055) ** 2.4); return .2126 * r + .7152 * g + .0722 * b; };
const ratio = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + .05) / (y + .05); };

// [text token, background token]: the pairs the screens use for text
const PAIRS = [
  ...['ink', 'ink-2', 'ink-3', 'muted'].flatMap(t => ['bg', 'surface', 'surface-2'].map(b => [t, b])),
  ['accent-text', 'bg'], ['accent-text', 'surface'], ['accent-text', 'accent-soft'],
  ['accent-ink', 'accent'], ['arena', 'bg'], ['arena', 'surface'], ['arena-ink', 'arena'],
  ['go', 'bg'], ['go', 'surface'], ['bad', 'bg'], ['bad', 'surface'], ['xp', 'bg'], ['xp', 'surface'], ['streak', 'bg'], ['medium', 'surface'],
];
for (const [name, T] of [['dark', dark], ['mobile', mobile]]) {
  test(`${name} palette: text tokens reach 4.5:1`, () => {
    assert.deepEqual(PAIRS.flat().filter(t => !T[t]), [], 'every checked token is defined');
    const bad = PAIRS.filter(([t, b]) => ratio(T[t], T[b]) < 4.5).map(([t, b]) => `${t} on ${b}: ${ratio(T[t], T[b]).toFixed(2)}`);
    assert.deepEqual(bad, []);
  });
}
