// [text](lesson:<course key>:<id>) → that lesson's page, e.g. [Evaluating RAG](lesson:aieng:10.04).
// Lessons link by id, so a moved module or renamed slug never breaks them; an unknown id fails the build.
import fs from 'node:fs';
import path from 'node:path';
import { visit } from 'unist-util-visit';

const ROOT = path.resolve(process.cwd(), 'src/content/courses');   // cwd is site/ (bundled copies can't use import.meta.url)
const BASE = (import.meta.env?.BASE_URL ?? '/self-learning-tracks/').replace(/\/$/, '');
let map;   // ponytail: built once per process; restart `astro dev` after adding a lesson

function build() {
  map = {};
  for (const dir of fs.readdirSync(ROOT)) {
    const cj = path.join(ROOT, dir, 'course.json');
    if (!fs.existsSync(cj)) continue;
    const c = JSON.parse(fs.readFileSync(cj, 'utf8')), mods = Object.fromEntries(c.modules.map(m => [m.n, m.slug]));
    for (const f of fs.readdirSync(path.join(ROOT, dir, 'lessons'))) {
      const fm = fs.readFileSync(path.join(ROOT, dir, 'lessons', f), 'utf8').split('\n---', 1)[0];
      const get = k => fm.match(new RegExp(`^${k}: "?([^"\\n]+)"?`, 'm'))?.[1];
      map[`${c.key}:${get('id')}`] = `${BASE}/library/${dir}/${mods[get('module')]}/${get('slug')}/`;
    }
  }
}

// "aieng:10.04" → its page URL (with the base path); throws on an unknown lesson.
export function lessonHref(ref, where = '') {
  map ??= (build(), map);
  if (!map[ref]) throw new Error(`lesson:${ref}: no such lesson ${where}`);
  return map[ref];
}

export default function remarkLessonLinks() {
  return (tree, file) => visit(tree, 'link', node => {
    if (node.url.startsWith('lesson:')) node.url = lessonHref(node.url.slice(7), `(${file.path})`);
  });
}
