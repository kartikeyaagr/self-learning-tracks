// Progress helpers on top of the single site-wide store (store.js).
import { ns, save as saveAll, all } from './store.js';
export const key = id => id.replace('.', '-');
export const load = k => { const st = ns(k); ['done', 'at', 'notes', 'grades', 'rev', 'act'].forEach(x => st[x] = st[x] || {}); return st; };
export const save = () => saveAll();
export const now = () => new Date().toISOString();
const pad = n => String(n).padStart(2, '0');
export const dk = d => { const x = new Date(d); return `${x.getFullYear()}-${pad(x.getMonth() + 1)}-${pad(x.getDate())}`; };
export const today = () => dk(Date.now());
export const addDays = (d, n) => { const x = new Date(d + 'T12:00:00'); x.setDate(x.getDate() + n); return dk(x); };
export const bump = st => { const d = today(); st.act = st.act || {}; st.act[d] = (st.act[d] || 0) + 1; };
const GB = {1: 3, 2: 8, 3: 15};
const xpOf = st => Object.values(st.done || {}).filter(Boolean).length * 10 + Object.values(st.grades || {}).reduce((a, g) => a + (GB[g?.g] || 0), 0)
  + Object.values(st.rev || {}).reduce((a, r) => a + (r?.n || 0), 0) * 5;
function days(st, c = {}) {
  Object.entries(st.at || {}).forEach(([k, t]) => { if ((st.done || {})[k] && t) c[dk(t)] = (c[dk(t)] || 0) + 1; });
  Object.entries(st.act || {}).forEach(([d, n]) => c[d] = (c[d] || 0) + (+n || 0));
  return c;
}
const LV = [0, 150, 400, 800, 1400, 2200, 3200, 4400, 6000, 8000];
const LVN = ['Novice', 'Apprentice', 'Learner', 'Practitioner', 'Builder', 'Engineer', 'Senior', 'Staff', 'Principal', 'Fellow'];
// Site-wide: XP and streak add up across every course and the Practice Lab.
export function stats() {
  const A = all(), c = {}; let xp = 0;
  for (const st of Object.values(A)) { if (st && typeof st === 'object') { xp += xpOf(st); days(st, c); } }
  let d = today(); if (!c[d]) d = addDays(d, -1); let streak = 0; while (c[d]) { streak++; d = addDays(d, -1); }
  let i = 0; while (i < LV.length - 1 && xp >= LV[i + 1]) i++;
  return {xp, streak, level: i + 1, name: LVN[i], days: c, today: c[today()] || 0};
}
