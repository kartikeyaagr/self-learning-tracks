// node --test src/lib/   (npm test). Checks the numbers the exemplar lesson's steps quote.
import test from 'node:test';
import assert from 'node:assert/strict';
import { softmax, SCORES, WORDS } from './softmax-temperature.js';

const pct = (T, w) => Math.round(softmax(SCORES, T)[WORDS.indexOf(w)] * 100);

test('softmax is stable and sums to 1', () => {
  assert.deepEqual(softmax([1000, 999, 998]).map(x => +x.toFixed(2)), [0.67, 0.24, 0.09]);
  assert.ok(Math.abs(softmax(SCORES, 0.25).reduce((a, b) => a + b) - 1) < 1e-12);
});

test('the lesson quotes these weights', () => {
  assert.equal(pct(1, 'animal'), 43);          // step 4: "about 43%"
  assert.equal(pct(3, 'animal'), 17);          // step 5: "about 17%", still the largest
  assert.ok(pct(3, 'animal') > pct(3, 'street'));
  assert.ok(pct(0.25, 'animal') > 90);         // low T piles the weight onto the top score
});
