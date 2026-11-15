// Client helpers shared by the Unroll screens: route data, the chosen destination, URLs, escaping, review due count.
import { all, ns, save } from './store.js';
import { due } from './review.js';

const B = () => window.__base;
const cache = {};
const json = p => (cache[p] ||= fetch(`${B()}/${p}`).then(r => r.json()));
// routes.json (practice/src/build.py): {stages, concepts, destinations, sets}, with lookups added.
export const routes = () => json('practice/routes.json').then(R => (R.byId ||= Object.fromEntries(R.concepts.map(c => [c.id, c])),
  R.dest ||= Object.fromEntries(R.destinations.map(d => [d.id, d])), R));
export const dueCards = () => json('cards.json').then(cards => due(cards)).catch(() => []);

// The chosen destination lives in the "game" namespace of the progress store.
export const getRoute = () => ns('game').route || null;
export const setRoute = id => { ns('game').route = id; save(); };
export const state = () => all();

export const lessonHref = (cid, slug) => `${B()}/learn/${cid}/${slug}/`;
export const conceptHref = cid => `${B()}/learn/${cid}/`;
export const problemHref = id => `${B()}/arena/p/${id}/`;
export const href = p => `${B()}${p}`;
export const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
export const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;

// Course colour token for a course folder id (tokens.css --course-*).
export const courseVar = id => `var(--course-${{'ml-maths-course': 'ml-maths', 'ml-toolkit': 'ml-toolkit', 'machine-learning': 'machine-learning',
  'deep-learning': 'deep-learning', 'ai-research-track-web': 'ai-research', 'ai-engineering-course': 'ai-engineering',
  'inference-eng-course': 'inference-engineering', 'system-design-101': 'system-design', 'ml-system-design': 'ml-system-design'}[id]})`;

// Locked things shake once when tapped (motion.css m-shake); call on any container.
export function shakeLocked(root) {
  root.addEventListener('click', e => { const el = e.target.closest('[data-locked]'); if (!el) return;
    e.preventDefault(); el.classList.remove('m-shake'); void el.offsetWidth; el.classList.add('m-shake'); });
}
