// Practice Lab data, written by practice/src/build.py into public/practice/ (verified problems, plans, papers, rounds).
import fs from 'node:fs';
import path from 'node:path';
import { u } from './url.mjs';
const DIR = path.resolve(process.cwd(), 'public/practice');
const read = (...p) => JSON.parse(fs.readFileSync(path.join(DIR, ...p), 'utf8'));
export const IDX = read('index.json');
export const ALL = IDX.topics.flatMap(m => m.lessons.map((l, i) => ({...l, mod: m.n, topic: m.name, tslug: m.slug, idx: i})));
export const BY = Object.fromEntries(ALL.map(l => [l.id, l]));
export const problem = slug => read('data', `${slug}.json`);
export const quiz = id => { try { return read('quiz', `${id}.json`); } catch { return null; } };
export const RT = {python: 'Python', numpy: 'NumPy', pandas: 'Pandas', sql: 'SQL', torch: 'PyTorch · Colab', jax: 'JAX · Colab', cuda: 'CUDA (Numba) · Colab', triton: 'Triton · Colab'};
export const ACCENT = {l: '#C03A73', d: '#F28DB8', ink_d: '#2A0A18', soft_l: '#FCEEF4', soft_d: '#2E1621', line_l: '#F1C6D8', line_d: '#4F2638'};
export const pHref = id => u(`/practice/p/${id}/`);
export const counts = ls => ['Easy', 'Medium', 'Hard'].map(d => [d, ls.filter(l => l.d === d).length]).filter(x => x[1]);
export const planItems = pl => pl.sections.flatMap(s => s.problems).map(id => BY[id]).filter(Boolean);
export const buildItems = b => [...b.steps.map(s => s.problem), b.capstone].map(id => BY[id]).filter(Boolean);
