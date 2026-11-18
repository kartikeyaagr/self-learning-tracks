import { test } from 'node:test';
import assert from 'node:assert/strict';
import links from '../plugins/remark-lesson-links.mjs';

const run = url => { const t = {type: 'root', children: [{type: 'link', url, children: []}]}; links()(t, {path: 't'}); return t.children[0].url; };
test('lesson: links resolve to the lesson page and unknown ids fail', () => {
  assert.match(run('lesson:ieng:08.03'), /^\/self-learning-tracks\/library\/inference-eng-course\/[^/]+\/pagedattention\/$/);
  assert.equal(run('https://example.com/'), 'https://example.com/');
  assert.throws(() => run('lesson:aieng:99.99'), /no such lesson/);
});
