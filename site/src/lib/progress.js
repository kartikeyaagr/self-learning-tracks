// Progress helpers on top of the single site-wide store (store.js).
import { ns, save as saveAll, all } from './store.js';
import { dk, addDays, events, summary } from './game.js';
export { dk, addDays };
export const key = id => id.replace('.', '-');
export const load = k => { const st = ns(k); ['done', 'at', 'notes', 'grades', 'rev', 'act'].forEach(x => st[x] = st[x] || {}); return st; };
export const save = () => saveAll();
export const now = () => new Date().toISOString();
export const today = () => dk(Date.now());
export const bump = st => { const d = today(); st.act = st.act || {}; st.act[d] = (st.act[d] || 0) + 1; };
// Site-wide XP, level, rank, streak and daily goal under the Unroll rules (game.js), across every course and the Arena.
export function stats() {
  const s = summary(events(all()), today());
  return {...s, level: s.level.n, lv: s.level};
}
