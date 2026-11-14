// Browser progress, in the same shape and localStorage keys as the original course pages, so nothing is lost.
export const key = id => id.replace('.', '-');
export function load(k) { try { const v = JSON.parse(localStorage.getItem(k) || '{}'); return v && typeof v === 'object' ? v : {}; } catch { return {}; } }
export function save(k, st) { try { localStorage.setItem(k, JSON.stringify(st)); } catch {} }
export const now = () => new Date().toISOString();
const pad = n => String(n).padStart(2, '0');
export const dk = d => { const x = new Date(d); return `${x.getFullYear()}-${pad(x.getMonth() + 1)}-${pad(x.getDate())}`; };
const addDays = (d, n) => { const x = new Date(d + 'T12:00:00'); x.setDate(x.getDate() + n); return dk(x); };
const GB = {1: 3, 2: 8, 3: 15};
export function stats(st, LV, LVN) {
  const done = st.done || {}, at = st.at || {}, rev = st.rev || {}, c = {};
  Object.entries(at).forEach(([k, t]) => { if (done[k] && t) c[dk(t)] = (c[dk(t)] || 0) + 1; });
  Object.entries(st.act || {}).forEach(([d, n]) => c[d] = (c[d] || 0) + (+n || 0));
  let d = dk(Date.now()); if (!c[d]) d = addDays(d, -1); let streak = 0; while (c[d]) { streak++; d = addDays(d, -1); }
  const n = Object.values(done).filter(Boolean).length;
  const xp = n * 10 + Object.values(st.grades || {}).reduce((a, g) => a + (GB[g?.g] || 0), 0) + Object.values(rev).reduce((a, r) => a + (r?.n || 0), 0) * 5;
  let i = 0; while (i < LV.length - 1 && xp >= LV[i + 1]) i++;
  return {n, xp, streak, level: i + 1, name: LVN[i], days: c};
}
