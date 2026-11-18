// Spaced review across the whole site: lessons you finished (with a self-check question) and problems you solved.
// Cards come back after 1, 3, 7, 14, then 30 days, the schedule the course pages used.
import { all, save } from './store.js';
import { today, addDays, now, bump, dk } from './progress.js';
export const BOX = [0, 1, 3, 7, 14, 30];
// cards: {namespace: [progress keys that have a card]} from /cards.json; practice problems are always cards.
export function due(cards = null) {
  const A = all(), t = today(), out = [];
  for (const [ns, st] of Object.entries(A)) {
    if (!st || typeof st !== 'object') continue;
    const ok = ns === 'practice' ? null : cards ? new Set(cards[ns] || []) : null;
    if (cards && ns !== 'practice' && !cards[ns]) continue;
    for (const [k, d] of Object.entries(st.done || {})) {
      if (!d || (ok && !ok.has(k))) continue;
      const r = (st.rev || {})[k];
      if (!r || !r.due || r.due <= t) out.push([ns, k]);
    }
    // Placement "review" modules: their cards join the deck without the lesson being done, PLACE_NEW new ones a day.
    const mods = st.place?.modules, rev = st.rev || {};
    if (!ok || !mods) continue;
    const keys = [...ok].filter(k => mods[k.slice(0, 2)] === 'review' && !st.done?.[k]).sort();
    let fresh = PLACE_NEW - keys.filter(k => rev[k]?.n === 1 && rev[k].at && dk(rev[k].at) === t).length;
    for (const k of keys) {
      const r = rev[k];
      if (r ? !r.due || r.due <= t : fresh-- > 0) out.push([ns, k]);
    }
  }
  return out;
}
const PLACE_NEW = 10;
// v: 1 again (back tomorrow), 2 hard (same interval), 3 got it (next interval), 4 easy (skips an interval). miss counts "again" grades;
// two make the card a refresher on the Learn path (game.js refreshers), and doing the refresher resets it.
export function rate(ns, k, v) {
  const st = all()[ns]; st.rev = st.rev || {};
  const r = st.rev[k] || {box: 0, n: 0}, b = v === 1 ? 0 : v === 2 ? Math.max(1, r.box || 0) : Math.min((r.box || 0) + (v === 4 ? 2 : 1), BOX.length - 1);
  st.rev[k] = {box: b, due: addDays(today(), Math.max(1, BOX[b])), n: (r.n || 0) + 1, at: now(), miss: (r.miss || 0) + (v === 1)};
  bump(st); save();
}
