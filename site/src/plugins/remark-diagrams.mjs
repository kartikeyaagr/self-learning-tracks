// ```dg-<kind> caption``` code blocks → the house diagram markup (static SVG/HTML), rendered at build time.
// Interactive kinds (widget, anim) render a placeholder that a client island fills in.
import { visit } from 'unist-util-visit';
import { diagram } from '../lib/diagrams.gen.mjs';
import { esc, inl } from '../lib/inline.mjs';

export default function remarkDiagrams() {
  return tree => {
    visit(tree, 'code', (node, index, parent) => {
      if (!node.lang || !node.lang.startsWith('dg-')) return;
      const kind = node.lang.slice(3), cap = (node.meta || '').trim();
      let html;
      if (kind === 'raw') {
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
