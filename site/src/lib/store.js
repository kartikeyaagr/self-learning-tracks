// One progress store for the whole site: localStorage "tracks" = {v: 1, ns: {<course key or "practice">: {...}}}.
// Each namespace keeps the shape the original course pages used (done, at, notes, grades, rev, act, …).
// First visit: the old per-course keys are copied in once and left untouched, so nobody loses progress.
const KEY = 'tracks';
const OLD = ['mlmaths', 'mltk', 'mlcore', 'dlcore', 'mlsd', 'air-fanout-v1', 'aieng', 'ieng', 'sdsd', 'practice'];
const get = k => { try { return JSON.parse(localStorage.getItem(k) || 'null'); } catch { return null; } };
let S = get(KEY);
if (!S || S.v !== 1) {
  S = {v: 1, ns: {}};
  for (const k of OLD) { const v = get(k); if (v && typeof v === 'object') S.ns[k] = v; }
  try { localStorage.setItem(KEY, JSON.stringify(S)); } catch {}
}
export const ns = name => (S.ns[name] = S.ns[name] && typeof S.ns[name] === 'object' ? S.ns[name] : {});   // live object
export const all = () => S.ns;
export function save() { try { localStorage.setItem(KEY, JSON.stringify(S)); } catch {} if (typeof window !== 'undefined') dispatchEvent(new Event('tracks:save')); }
// For course code that reads and writes its own JSON blob (AI Research's notebook blocks).
export const raw = { get: name => JSON.stringify(ns(name)), set: (name, str) => { try { S.ns[name] = JSON.parse(str); save(); } catch {} } };
if (typeof window !== 'undefined') window.__store = { ns, save, raw };
