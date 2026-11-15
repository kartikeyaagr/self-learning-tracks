// Unroll game engine: XP, levels, Arena rank, streak, daily goal, quests, stars, weekly bonus, refreshers, route state.
// Pure functions over the progress store's namespaces (store.js all()); nothing here reads or writes localStorage.
// XP is derived from what the pages already record (finished lessons, solves, hints, review boxes), so old progress
// counts under these rules and undoing a lesson takes its XP back. Rules: design/unroll/HANDOFF.md → Gamification.
//
// State read here, by namespace:
//   <course key>  done[k], at[k]            finished lessons (k = lesson id with "-", e.g. "01-02")
//                 steps[k] = [iso, …]       finished lesson steps (written by the lesson player)
//                 rev[k] = {box, due, n, at, miss}
//   practice      done[slug], at[slug], hints[slug] (count), hintAt[slug] = [iso, …], sol[slug], grades[slug]
//                 solve[slug] = {d, theory, nohints, ms, potd}   recorded at the first solve
//   game          clear[day] = iso (review deck cleared), checkpoints[concept] = {best, n, at}

export const XP = {step: 5, lesson: 15, solve: {Easy: 20, Medium: 30, Hard: 40}, theory: 5, nohints: 10, par: 10,
  hint: -5, potd: 10, weekly: 60, deck: 20};
export const GOAL = 50;                      // daily goal; meeting it keeps the streak
export const WEEK_SOLVES = 5;                // weekly target → XP.weekly
export const PAR = {Easy: 10, Medium: 20, Hard: 30};   // minutes; the problem's listed time
export const CHECKPOINT = {n: 6, pass: 5};   // questions, and correct answers needed to pass (or test out)
export const REFRESHER = {misses: 2, minutes: 2};
const DIFF = {1: 'Easy', 2: 'Medium', 3: 'Hard'};   // legacy: practice grades[slug].g was the difficulty

// Dates are local calendar days, "YYYY-MM-DD".
const pad = n => String(n).padStart(2, '0');
export const dk = d => { const x = new Date(d); return `${x.getFullYear()}-${pad(x.getMonth() + 1)}-${pad(x.getDate())}`; };
export const addDays = (d, n) => { const x = new Date(d + 'T12:00:00'); x.setDate(x.getDate() + n); return dk(x); };
export const weekOf = d => addDays(d, -((new Date(d + 'T12:00:00').getDay() + 6) % 7));   // Monday
const obj = x => x && typeof x === 'object' ? x : {};

// Every XP award as {t, d, xp, kind, ref, arena}. t/d are null for legacy awards with no timestamp:
// they count toward totals but not toward any day.
export function events(A) {
  const out = [], P = obj(A.practice), G = obj(A.game);
  const add = (t, xp, kind, ref, arena = false) => out.push({t: t || null, d: t ? dk(t) : null, xp, kind, ref, arena});
  for (const [ns, st] of Object.entries(A)) {
    if (ns === 'practice' || ns === 'game' || !st || typeof st !== 'object') continue;
    for (const [k, v] of Object.entries(obj(st.done))) if (v) add(obj(st.at)[k], XP.lesson, 'lesson', `${ns}:${k}`);
    for (const [k, ts] of Object.entries(obj(st.steps))) for (const t of ts || []) add(t, XP.step, 'step', `${ns}:${k}`);
  }
  const at = obj(P.at), solves = [];
  for (const [k, v] of Object.entries(obj(P.done))) {
    if (!v) continue;
    const r = obj(obj(P.solve)[k]), t = at[k], d = r.d || DIFF[obj(obj(P.grades)[k]).g] || 'Easy';
    add(t, XP.solve[d], 'solve', k, true); solves.push(t);
    if (r.theory) add(t, XP.theory, 'star:theory', k, true);
    if (r.d ? r.nohints : !(obj(P.hints)[k] > 0) && !obj(P.sol)[k]) add(t, XP.nohints, 'star:nohints', k, true);
    if (r.ms != null && r.ms <= PAR[d] * 60000) add(t, XP.par, 'star:par', k, true);
    if (r.potd) add(t, XP.potd, 'potd', k, true);
  }
  for (const [k, n] of Object.entries(obj(P.hints))) {   // stamped hints at their time, older ones at the solve (if any)
    const ts = obj(P.hintAt)[k] || [], old = Math.max(0, n - ts.length);
    for (let i = 0; i < n; i++) add(i < old ? at[k] : ts[i - old], XP.hint, 'hint', k, true);
  }
  const byWeek = {};                                     // weekly bonus lands on the week's 5th solve
  for (const t of solves.filter(Boolean).sort()) (byWeek[weekOf(dk(t))] ||= []).push(t);
  for (const [w, ts] of Object.entries(byWeek)) if (ts.length >= WEEK_SOLVES) add(ts[WEEK_SOLVES - 1], XP.weekly, 'weekly', w, true);
  for (const [d, t] of Object.entries(obj(G.clear))) add(t, XP.deck, 'deck', d);
  return out;
}

// Level n starts at 50·n·(n−1) total XP; Arena rank r at 25·r·(r−1) Arena XP (mockup: Lv 7 at 2,340; rank 7 at 1,180).
export const tier = (x, k) => { let n = 1; while (x >= k * (n + 1) * n) n++; return {n, at: k * n * (n - 1), next: k * (n + 1) * n}; };
export const level = xp => tier(xp, 50);
export const rank = xp => tier(xp, 25);

export function summary(evs, day) {
  let xp = 0, axp = 0; const days = {};
  for (const e of evs) { xp += e.xp; if (e.arena) axp += e.xp; if (e.d) days[e.d] = (days[e.d] || 0) + e.xp; }
  xp = Math.max(0, xp); axp = Math.max(0, axp);
  const met = d => (days[d] || 0) >= GOAL;
  let d = met(day) ? day : addDays(day, -1), streak = 0;   // today still counts as "kept" until it ends
  while (met(d)) { streak++; d = addDays(d, -1); }
  const wk = weekOf(day);
  return {xp, arenaXp: axp, level: level(xp), rank: rank(axp), days, today: days[day] || 0, goal: GOAL, goalMet: met(day), streak,
    week: {start: wk, met: [0, 1, 2, 3, 4, 5, 6].map(i => met(addDays(wk, i))),
      solves: evs.filter(e => e.kind === 'solve' && e.d && weekOf(e.d) === wk).length, target: WEEK_SOLVES}};
}

// Today's quests. due = review cards still due now.
export function quests(A, evs, day, due) {
  const n = kind => evs.filter(e => e.kind === kind && e.d === day).length;
  let reviewed = 0;
  for (const st of Object.values(A)) for (const r of Object.values(obj(obj(st).rev))) if (r?.at && dk(r.at) === day) reviewed++;
  const cleared = !!obj(obj(A.game).clear)[day];
  return [
    {id: 'lesson', label: 'Finish one lesson', n: Math.min(n('lesson'), 1), of: 1},
    {id: 'solve', label: 'Solve 2 Arena problems', n: Math.min(n('solve'), 2), of: 2},
    {id: 'deck', label: 'Clear your review deck', n: cleared ? Math.max(reviewed, 1) : reviewed, of: cleared ? Math.max(reviewed, 1) : reviewed + due, xp: XP.deck},
  ].map(q => ({...q, done: q.of > 0 && q.n >= q.of}));
}

// Route state. ref = "<course key>:<lesson id>" (concepts.json). A concept is done when every lesson is done or its
// checkpoint was passed (testing out); current = first unfinished required concept; locked = some need unfinished.
export const lessonDone = (A, ref) => { const [ns, id] = ref.split(':'); return !!obj(obj(A[ns]).done)[id.replace('.', '-')]; };
export const checkpointPassed = (A, cid) => (obj(obj(A.game).checkpoints)[cid]?.best || 0) >= CHECKPOINT.pass;
export function routeState(A, route, byId, optional = []) {
  const done = {};
  for (const cid of route) done[cid] = checkpointPassed(A, cid) || byId[cid].lessons.every(r => lessonDone(A, r));
  const current = route.find(cid => !done[cid] && !optional.includes(cid)) ?? null;
  return route.map(cid => {
    const c = byId[cid], n = c.lessons.filter(r => lessonDone(A, r)).length;
    const status = done[cid] ? 'done' : cid === current ? 'current' : (c.needs || []).every(x => done[x] ?? true) ? 'open' : 'locked';
    return {id: cid, status, optional: optional.includes(cid), lessons: n, of: c.lessons.length};
  });
}

// Review cards missed (graded "again") twice become 2-minute refreshers on the Learn path, placed before the first
// unfinished route concept that needs the card's concept (else before the current concept). Doing one resets miss.
export function refreshers(A, route, byId, optional = []) {
  const state = routeState(A, route, byId, optional), open = state.filter(s => s.status !== 'done').map(s => s.id);
  const owner = {};
  for (const c of Object.values(byId)) { c.lessons.forEach(r => owner[r] = c.id); (c.problems || []).forEach(p => owner['practice:' + p] = c.id); }
  const out = [];
  for (const [ns, st] of Object.entries(A)) for (const [k, r] of Object.entries(obj(obj(st).rev))) {
    if ((r?.miss || 0) < REFRESHER.misses) continue;
    const ref = ns === 'practice' ? `practice:${k}` : `${ns}:${k.replace('-', '.')}`, c0 = owner[ref];
    const at = open.find(cid => c0 && (byId[cid].needs || []).includes(c0)) ?? state.find(s => s.status === 'current')?.id;
    if (at) out.push({at, ns, k, ref, concept: c0 ?? null, minutes: REFRESHER.minutes});
  }
  return out;
}

// A set's boss round opens when all but one of its problems are solved (8 of 9).
export const bossOpen = (A, problems) => problems.filter(p => obj(obj(A.practice).done)[p]).length >= problems.length - 1;

// Problem of the day: index into the full problem list (practice index.json topics → lessons, in order).
export const dailyIndex = (day, n) => { let h = 0; for (const c of day) h = (h * 31 + c.charCodeAt(0)) >>> 0; return h % n; };

// What a first solve earns, for the toast: [{label, xp}].
export function solveAwards(d, r) {
  return [['Solved', XP.solve[d]], r.theory && ['Theory first', XP.theory], r.nohints && ['No hints', XP.nohints],
    r.ms != null && r.ms <= PAR[d] * 60000 && ['Under par', XP.par], r.potd && ['Problem of the day', XP.potd]]
    .filter(Boolean).map(([label, xp]) => ({label, xp}));
}
