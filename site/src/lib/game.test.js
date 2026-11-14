// node --test src/lib/   (npm test)
import test from 'node:test';
import assert from 'node:assert/strict';
import * as G from './game.js';

const T = (day, h = 12) => `${day}T${pad(h)}:00:00`;   // local time, so dk() returns `day`
const pad = n => String(n).padStart(2, '0');
const xpOf = A => G.summary(G.events(A), '2026-11-15');

test('levels and ranks match the mockup', () => {
  assert.deepEqual(G.level(2340), {n: 7, at: 2100, next: 2800});
  assert.deepEqual(G.rank(1180), {n: 7, at: 1050, next: 1400});   // 220 to rank 8
  assert.equal(G.level(0).n, 1); assert.equal(G.level(99).n, 1); assert.equal(G.level(100).n, 2);
});

test('lessons and steps; undoing a lesson removes its XP', () => {
  const A = {dlcore: {done: {'01-01': true, '01-02': true}, at: {'01-01': T('2026-11-15'), '01-02': T('2026-11-14')},
    steps: {'01-03': [T('2026-11-15'), T('2026-11-15')]}}};
  const s = xpOf(A);
  assert.equal(s.xp, 15 + 15 + 10); assert.equal(s.today, 25); assert.equal(s.arenaXp, 0);
  delete A.dlcore.done['01-01'];
  assert.equal(xpOf(A).xp, 25);
});

test('first solve, stars, hints and problem of the day', () => {
  const t = T('2026-11-15');
  const A = {practice: {done: {a: true, b: true, c: true}, at: {a: t, b: t, c: t},
    solve: {a: {d: 'Hard', theory: true, nohints: true, ms: 29 * 60000, potd: true}, b: {d: 'Medium', nohints: false, ms: 21 * 60000}},
    hints: {b: 2, z: 1}, hintAt: {b: [t, t]}, grades: {c: {g: 2}}}};
  // a: 40 + 5 + 10 + 10 + 10 · b: 30 − 2 hints · c: legacy Medium (grades) + legacy no-hints star · z: hint, never solved
  const s = xpOf(A);
  assert.equal(s.xp, 75 + 30 - 10 + 40 - 5);
  assert.equal(s.arenaXp, s.xp);
  assert.equal(s.today, 75 + 20 + 40);                   // z's legacy hint has no date
  assert.deepEqual(G.solveAwards('Hard', A.practice.solve.a).map(x => x.xp), [40, 5, 10, 10, 10]);
});

test('legacy solve: viewing the solution loses the no-hints star; unknown difficulty counts as Easy', () => {
  const A = {practice: {done: {a: true}, at: {a: T('2026-11-15')}, sol: {a: true}}};
  assert.equal(xpOf(A).xp, 20);
});

test('total XP never goes below zero', () => assert.equal(xpOf({practice: {hints: {a: 3}}}).xp, 0));

test('streak counts days with the daily goal met; today is kept until it ends', () => {
  const steps = days => ({dlcore: {steps: {x: days.flatMap(d => Array(10).fill(T(d)))}}});   // 10 steps = 50 XP
  assert.equal(xpOf(steps(['2026-11-13', '2026-11-14'])).streak, 2);                          // today not met yet
  assert.equal(xpOf(steps(['2026-11-13', '2026-11-14', '2026-11-15'])).streak, 3);
  assert.equal(xpOf(steps(['2026-11-12', '2026-11-14'])).streak, 1);
  const s = xpOf(steps(['2026-11-10', '2026-11-15']));   // Monday and Sunday of the same week
  assert.equal(s.week.start, '2026-11-09');
  assert.deepEqual(s.week.met, [false, true, false, false, false, false, true]);
});

test('weekly bonus lands on the 5th solve of a week', () => {
  const days = ['2026-11-09', '2026-11-10', '2026-11-11', '2026-11-12', '2026-11-13', '2026-11-16'];
  const P = {done: {}, at: {}, solve: {}};
  days.forEach((d, i) => { P.done['p' + i] = true; P.at['p' + i] = T(d); P.solve['p' + i] = {d: 'Easy', nohints: false}; });
  const w = G.events({practice: P}).filter(e => e.kind === 'weekly');
  assert.deepEqual(w.map(e => [e.d, e.xp]), [['2026-11-13', 60]]);
});

test('quests and the deck-clear bonus', () => {
  const day = '2026-11-15', t = T(day);
  const A = {dlcore: {done: {'01-01': true}, at: {'01-01': t}, rev: {'01-01': {at: t}, '01-02': {at: t}}},
    practice: {done: {a: true}, at: {a: t}, solve: {a: {d: 'Easy', nohints: false}}}};
  let q = G.quests(A, G.events(A), day, 3);
  assert.deepEqual(q.map(x => [x.n, x.of, x.done]), [[1, 1, true], [1, 2, false], [2, 5, false]]);
  A.game = {clear: {[day]: t}};
  q = G.quests(A, G.events(A), day, 0);
  assert.equal(q[2].done, true);
  assert.equal(G.summary(G.events(A), day).xp, 15 + 20 + 20);
});

const C = {
  la: {id: 'la', lessons: ['mlmaths:03.01'], needs: []},
  att: {id: 'att', lessons: ['aieng:04.02', 'aieng:04.05'], needs: ['la']},
  tok: {id: 'tok', lessons: ['dlcore:06.01'], needs: []},
  kv: {id: 'kv', lessons: ['ieng:03.04'], needs: ['att'], problems: ['kv-p']},
  pa: {id: 'pa', lessons: ['ieng:08.01'], needs: ['kv']},
};
const route = ['la', 'att', 'tok', 'kv', 'pa'];

test('route state: done, current, open, locked, optional; checkpoint tests out', () => {
  const A = {mlmaths: {done: {'03-01': true}}, aieng: {done: {'04-02': true}}};
  const st = () => G.routeState(A, route, C, ['tok']).map(s => s.status);
  assert.deepEqual(st(), ['done', 'current', 'open', 'locked', 'locked']);
  A.game = {checkpoints: {att: {best: 5, n: 6}}};
  assert.deepEqual(st(), ['done', 'done', 'open', 'current', 'locked']);
  A.game.checkpoints.att.best = 4;
  assert.equal(st()[1], 'current');
});

test('refreshers go before the first unfinished concept that needs the missed card', () => {
  const A = {mlmaths: {done: {'03-01': true}}, aieng: {done: {'04-02': true, '04-05': true}, rev: {'04-02': {miss: 2}, '04-05': {miss: 1}}},
    practice: {rev: {'kv-p': {miss: 3}}}};
  const r = G.refreshers(A, route, C, ['tok']);
  assert.deepEqual(r.map(x => [x.ref, x.at]), [['aieng:04.02', 'kv'], ['practice:kv-p', 'pa']]);
});

test('boss opens at all but one solved; daily index is stable', () => {
  const A = {practice: {done: {a: true, b: true}}};
  assert.equal(G.bossOpen(A, ['a', 'b', 'c']), true);
  assert.equal(G.bossOpen(A, ['a', 'c', 'd']), false);
  let h = 0; for (const c of '2026-11-15') h = (h * 31 + c.charCodeAt(0)) >>> 0;   // the hash the daily page used
  assert.equal(G.dailyIndex('2026-11-15', 911), h % 911);
});
