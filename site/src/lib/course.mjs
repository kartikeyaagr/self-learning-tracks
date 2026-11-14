// Server-side helpers shared by the course pages.
import { getCollection } from 'astro:content';
import fs from 'node:fs';
import path from 'node:path';
import { inl } from './inline.mjs';

export const key = id => id.replace('.', '-');                    // lesson id "03.02" -> progress key "03-02"

const cache = {};
export async function courseData(courseId) {
  if (cache[courseId]) return cache[courseId];
  const course = (await getCollection('courses')).find(c => c.id === courseId).data;
  const order = Object.fromEntries(course.modules.map((m, i) => [m.n, i]));   // module codes aren't always numeric (IN, 00…)
  const lessons = (await getCollection('lessons', l => l.data.course === courseId))
    .sort((a, b) => order[a.data.module] - order[b.data.module] || a.data.id.localeCompare(b.data.id));
  const modules = course.modules.map(m => ({...m, lessons: lessons.filter(l => l.data.module === m.n)}));
  const mfile = path.resolve(process.cwd(), 'src/content/courses', courseId, 'math.json');
  const math = fs.existsSync(mfile) ? JSON.parse(fs.readFileSync(mfile, 'utf8')) : null;
  return (cache[courseId] = {course, lessons, modules, math});
}
export const lessonUrl = (courseId, mod, l) => `/courses/${courseId}/${mod.slug}/${l.data.slug}/`;
// Inline markdown for frontmatter text; exports keep maths as ⟦n⟧ references into the course's MathML table.
export const inlM = (s, math) => { const h = inl(s); return math ? h.replace(/⟦(\d+)⟧/g, (_, n) => math[+n] || '') : h; };

// Practice problems for a lesson: the lesson's own picks first, then problems that cite it (practice/links.json).
const PRACTICE = path.resolve(process.cwd(), '../practice');
const links = JSON.parse(fs.readFileSync(path.join(PRACTICE, 'links.json'), 'utf8'));
const prob = s => { try { const p = JSON.parse(fs.readFileSync(path.join(PRACTICE, 'data', s + '.json'), 'utf8')); return [s, p.t, p.d]; } catch { return null; } };
export function practiceFor(courseKey, lesson) {
  const out = lesson.data.practice.map(prob).filter(Boolean);
  for (const x of links[`${courseKey}:${lesson.data.id}`] || []) if (!out.some(o => o[0] === x[0])) out.push(x);
  return out;
}
// The Practice Lab moves to Astro in phase 3; until then link to the current one.
export const practiceHref = s => `${import.meta.env.PUBLIC_PRACTICE_BASE ?? '/practice/index.html'}#l${s}`;
