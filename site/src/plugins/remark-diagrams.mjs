// ```dg-<kind> caption``` code blocks → the house diagram markup (static SVG/HTML), rendered at build time.
// Interactive kinds (widget, anim) render a placeholder that a client island fills in.
import fs from 'node:fs';
import path from 'node:path';
import { visit } from 'unist-util-visit';
import { diagram } from '../lib/diagrams.gen.mjs';
import { esc, inl } from '../lib/inline.mjs';

// ~~~dg-fig <name>.svg + caption body: a plotted figure from the course's figs/ folder, inlined so its
// var(--ink, …) colours follow the theme. Ids are prefixed per figure so several on one page never collide.
function figure(file, name, cap) {
  const dir = file.path && file.path.match(/^(.*\/courses\/[^/]+)\//)?.[1];
  const f = dir && path.join(dir, 'figs', name);
  if (!f || !fs.existsSync(f)) throw new Error(`dg-fig: ${name} not found in ${dir ? dir + '/figs' : 'this file (no course folder)'} (${file.path})`);
  const p = name.replace(/\W/g, '_') + '-';
  const svg = fs.readFileSync(f, 'utf8').replace(/<\?xml[^>]*>|<!DOCTYPE[^>]*>|<metadata>[\s\S]*?<\/metadata>/g, '')
    .replace(/\bid="([^"]+)"/g, `id="${p}$1"`).replace(/url\(#([^)]+)\)/g, `url(#${p}$1)`).replace(/href="#([^"]+)"/g, `href="#${p}$1"`)
    .replace(/\s+/g, ' ').trim();
  return `<figure class="dg fig">${svg}${cap ? `<figcaption>${inl(cap)}</figcaption>` : ''}</figure>`;
}

export default function remarkDiagrams() {
  return (tree, file) => {
    visit(tree, 'code', (node, index, parent) => {
      if (!node.lang || !node.lang.startsWith('dg-')) return;
      const kind = node.lang.slice(3), cap = (node.meta || '').trim();
      let html;
      if (kind === 'fig') {
        html = figure(file, cap, node.value.trim().replace(/\s*\n\s*/g, ' '));
      } else if (kind === 'raw') {
        // A course-specific figure: keep the original source; the course's own code renders it in the browser.
        html = `<div class="dg-raw" data-src="${esc(node.value)}"></div>`;
      } else if (kind === 'anim') {
        const [id, ...rest] = cap.split('|'), c = rest.join('|').trim();
        html = `<figure class="dg anim" data-anim="${esc(id.trim())}"><div class="anim-stage"></div>${c ? `<figcaption>${inl(c)}</figcaption>` : ''}</figure>`;
      } else {
        const lines = node.value.split('\n').map(l => kind === 'tree' ? l.replace(/\s+$/, '') : l.trim()).filter(Boolean);
        html = diagram(kind, cap, lines);
      }
      parent.children[index] = {type: 'html', value: html};
    });
  };
}
