// Build-time helpers for the lesson player (pages/learn/[concept]/[lesson].astro).
// A lesson's steps come from its frontmatter `steps:` (content.config.ts), one JSON array on one line:
//   [{"kind": "explain" | "interact" | "question", "title": "…", "body": "Markdown", "widget": "softmax-temperature"?,
//     "question": {"q": "…", "options": ["…", "…", "…"], "answer": 0, "why": "shown when right", "hint": "shown when wrong"}?}]
// body is block Markdown (maths, lists, code); q, options, why and hint are inline Markdown. widget names a module in
// lib/widgets/ (export mount(el)). Feedback is the question's checked state, not a step of its own.
// A lesson without steps plays as "read" (the full lesson, figures included) then "check" (its self-check question,
// model answer and self-grade), when it has a question with a model answer.
import fs from 'node:fs';
import path from 'node:path';
import { md } from './md.mjs';

let R;
// routes.json (practice/src/build.py): concepts with lesson_links (slug, t) and problem_links (id, t, d).
export const routes = () => (R ??= JSON.parse(fs.readFileSync(path.resolve(process.cwd(), 'public/practice/routes.json'), 'utf8')));

const SHORT = 24;   // options longer than this stack vertically, and "Right: <option>" becomes plain "Right"
export async function buildSteps(L, I) {
  if (!L.steps?.length) return [{kind: 'read', title: 'Read the lesson'}, ...(L.question && L.answer ? [{kind: 'check', title: 'Check yourself'}] : [])];
  return Promise.all(L.steps.map(async s => {
    const q = s.question;
    return {kind: s.kind, title: s.title, html: await md(s.body), widget: s.widget || null, q: q && {
      q: I(q.q), options: q.options.map(I), answer: q.answer, why: I(q.why), hint: I(q.hint),
      long: q.options.some(o => o.length > SHORT), right: q.options[q.answer].length <= SHORT ? I(q.options[q.answer]) : null}};
  }));
}
