// Wrap tables in .tbl so they scroll on narrow screens and pick up the house table style.
import { visit } from 'unist-util-visit';
export default function rehypeTables() {
  return tree => visit(tree, 'element', (node, index, parent) => {
    if (node.tagName !== 'table' || parent?.properties?.className?.includes('tbl')) return;
    parent.children[index] = {type: 'element', tagName: 'div', properties: {className: ['tbl']}, children: [node]};
  });
}
