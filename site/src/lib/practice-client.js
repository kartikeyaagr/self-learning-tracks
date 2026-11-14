// Practice Lab, in the browser: fills in solved counts from the progress store.
//   [data-ids="a|b|c"]   .dn → solved count, .bar i → width, .okb shown when all solved, a[data-next] → first unsolved
//   [data-quiz="plan:si"] best quiz score    [data-round=id] last round score
import { load } from './progress.js';
export const st = load('practice');
['code', 'hints', 'sol', 'expl', 'quiz', 'rounds', 'live'].forEach(x => st[x] = st[x] || {});
export const isDone = id => !!st.done[id];
export const href = id => `${window.__base}/practice/p/${id}/`;
export function toast(html, big) {
  const t = document.createElement('div'); t.className = 'toast' + (big ? ' big' : ''); t.innerHTML = html;
  document.getElementById('toasts').appendChild(t); setTimeout(() => t.remove(), big ? 3200 : 2200);
}
document.querySelectorAll('[data-ids]').forEach(el => {
  const ids = el.dataset.ids.split('|').filter(Boolean), d = ids.filter(isDone).length, all = d === ids.length && d > 0;
  el.querySelectorAll('.dn').forEach(x => x.textContent = d);
  el.querySelectorAll('.bar i').forEach(x => x.style.width = (ids.length ? d / ids.length * 100 : 0) + '%');
  el.querySelectorAll('.okb').forEach(x => x.hidden = !all);
  el.querySelectorAll('a[data-next]').forEach(a => { const [s, c, again] = a.dataset.next.split('|');
    a.href = href(ids.find(i => !isDone(i)) || ids[0]); a.firstChild.textContent = (d ? (all ? again : c) : s) + ' → '; });
});
document.querySelectorAll('[data-quiz]').forEach(el => { const b = st.quiz[el.dataset.quiz]; if (b) { el.textContent = `Quiz · ${b.best}/${b.n}`; el.className = 'pill ' + (b.best === b.n ? 'ok' : 'ip'); } });
document.querySelectorAll('[data-round]').forEach(el => { const h = st.rounds[el.dataset.round] || [], l = h[h.length - 1];
  if (l) { el.hidden = false; el.textContent = `Last ${l.pct}%`; el.className = 'pill ' + (l.pct >= 70 ? 'ok' : 'ip'); } });
