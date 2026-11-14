// Which lessons have a review card (a self-check question with a model answer), by progress namespace.
import { getCollection } from 'astro:content';
export async function GET() {
  const keyOf = Object.fromEntries((await getCollection('courses')).map(c => [c.id, c.data.key]));
  const out = {};
  for (const l of await getCollection('lessons')) {
    if (!l.data.question || !l.data.answer) continue;
    (out[keyOf[l.data.course]] ||= []).push(l.data.id.replace('.', '-'));
  }
  return new Response(JSON.stringify(out));
}
