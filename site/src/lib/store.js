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
// Another tab saved, or this page came back from the back/forward cache: take the stored progress, updating each
// namespace object in place (pages hold live references from ns()), so the next save here can't overwrite it.
function refresh() {
  const fresh = get(KEY); if (!fresh || fresh.v !== 1) return;
  for (const name of new Set([...Object.keys(S.ns), ...Object.keys(fresh.ns)])) {
    const cur = S.ns[name], next = fresh.ns[name];
    if (cur && typeof cur === 'object' && next && typeof next === 'object') {
      for (const k of Object.keys(cur)) if (!(k in next)) cur[k] = cur[k] && typeof cur[k] === 'object' ? {} : undefined;   // keep the maps load() set up
      Object.assign(cur, next);
    }
    else if (next === undefined) delete S.ns[name]; else S.ns[name] = next;
  }
  dispatchEvent(new Event('tracks:sync')); dispatchEvent(new Event('tracks:save'));   // sync: progress changed elsewhere
}
if (typeof window !== 'undefined') {
  window.__store = { ns, save, raw };
  addEventListener('storage', e => { if (e.key === KEY) refresh(); });
  addEventListener('pageshow', e => { if (e.persisted) refresh(); });
}
