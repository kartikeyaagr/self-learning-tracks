// Every review card's content, fetched by the review page only: question and model answer as HTML, and the lesson URL.
import { getCollection } from 'astro:content';
import { courseData, lessonUrl, inlM } from '../lib/course.mjs';
export async function GET() {
  const out = {};
  for (const c of await getCollection('courses')) {
    const { course, modules, math } = await courseData(c.id);
    const cards = out[course.key] = {};
    for (const m of modules) for (const l of m.lessons) {
      if (!l.data.question || !l.data.answer) continue;
      cards[l.data.id.replace('.', '-')] = {t: l.data.title, c: course.title, q: inlM(l.data.question, math), s: inlM(l.data.answer, math), u: lessonUrl(c.id, m, l)};
    }
  }
  return new Response(JSON.stringify(out));
}
