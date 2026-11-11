const ALL = DATA.flatMap(m => m.lessons.map((l, i) => ({...l, mod: m.n, idx: i})));
const PARTS = [{t:'Foundations and retrieval', a:1, b:2}, {t:'How LLMs work', a:3, b:5}, {t:'Making models fast and small', a:6, b:7}, {t:'Reasoning and agents', a:8, b:9}];
const key = id => id.replace('.', '-');
const st = {done:{}, at:{}, notes:{}, flags:{}, reveal:{}, grades:{}, rev:{}, act:{}};
let db = null;
function lsGet(){ try { return JSON.parse(localStorage.getItem('aieng') || '{}'); } catch(e){ return {}; } }
function lsSet(){ try { const {done, at, notes, flags, grades, rev, act} = st; localStorage.setItem('aieng', JSON.stringify({done, at, notes, flags, grades, rev, act})); } catch(e){} }
Object.assign(st, lsGet()); st.reveal = {};
['done','at','notes','flags','grades','rev','act'].forEach(k => st[k] = st[k] || {});

const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const inl = s => esc(s).replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>').replace(/`(.+?)`/g,'<code>$1</code>').replace(/(^|[^*])\*([^*]+?)\*/g,'$1<em>$2</em>');
const plain = s => String(s).replace(/\*\*|`|\*/g, '');
const host = u => { try { return new URL(u).hostname.replace(/^www\./,''); } catch(e){ return ''; } };
const isDone = id => !!st.done[key(id)];
const modDone = m => m.lessons.filter(l => isDone(l.id)).length;
const words = l => (l.a || l.n.join(' ')).split(/\s+/).length;
const mins = l => Math.max(2, Math.ceil(words(l) / 180));
const pad = n => String(n).padStart(2, '0');
const dk = d => { const x = new Date(d); return x.getFullYear() + '-' + pad(x.getMonth()+1) + '-' + pad(x.getDate()); };
const today = () => dk(Date.now());
const addDays = (d, n) => { const x = new Date(d + 'T12:00:00'); x.setDate(x.getDate() + n); return dk(x); };
const now = () => new Date().toISOString();
const RM = matchMedia('(prefers-reduced-motion: reduce)').matches;
const CHECK = '<svg width="20" height="20" viewBox="0 0 20 20"><path d="M4.5 10.5l3.6 3.6 7.6-8.2" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const FLAME = '<svg width="14" height="16" viewBox="0 0 14 16" aria-hidden="true"><path d="M7 1c.5 2.6 4.5 4.4 4.5 8.5A4.5 4.5 0 0 1 2.5 9.5C2.5 7.6 3.6 6.4 4.4 5.6c.2 1.5.9 2.3 1.6 2.6C5.9 5.6 6.3 3 7 1z" fill="currentColor"/></svg>';

// ---------- game layer: XP, levels, streaks, badges ----------
const LV = [0, 60, 150, 300, 500, 750, 1050, 1400];
const LVN = ['Prompter', 'Builder', 'AI Engineer', 'Senior', 'Staff', 'Senior Staff', 'Principal', 'Research Engineer'];
const GB = {1: 3, 2: 8, 3: 15};
const doneCount = () => Object.values(st.done).filter(Boolean).length;
const revCount = () => Object.values(st.rev).reduce((a, r) => a + (r.n || 0), 0);
const xp = () => doneCount() * 10 + Object.values(st.grades).reduce((a, g) => a + (GB[g.g] || 0), 0) + revCount() * 5;
function level(x = xp()){ let i = 0; while (i < LV.length - 1 && x >= LV[i+1]) i++; const nx = LV[i+1]; return {i, name: LVN[i], x, from: LV[i], to: nx, pct: nx ? (x - LV[i]) / (nx - LV[i]) : 1}; }
function dayCounts(){
  const c = {};
  Object.entries(st.at).forEach(([k, t]) => { if (st.done[k] && t) { const d = dk(t); c[d] = (c[d] || 0) + 1; } });
  Object.entries(st.act).forEach(([d, n]) => c[d] = (c[d] || 0) + (n || 0));
  return c;
}
function streak(c = dayCounts()){
  let d = today(); if (!c[d]) d = addDays(d, -1); let n = 0;
  while (c[d]) { n++; d = addDays(d, -1); }
  return n;
}
function bestStreak(c = dayCounts()){
  let best = 0; Object.keys(c).forEach(d => { if (c[addDays(d, -1)]) return; let n = 0, x = d; while (c[x]) { n++; x = addDays(x, 1); } best = Math.max(best, n); });
  return best;
}
const todayLessons = () => Object.entries(st.at).filter(([k, t]) => st.done[k] && t && dk(t) === today()).length;
const GOAL = 2;
function badges(){
  const n = doneCount(), bs = bestStreak(), nail = Object.values(st.grades).filter(g => g.g === 3).length, rv = revCount(), full = DATA.some(m => modDone(m) === m.lessons.length);
  return [
    ['1', 'First step', 'Finish your first lesson', n >= 1],
    ['3d', 'On a roll', 'Learn 3 days in a row', bs >= 3],
    ['7d', 'Week streak', 'Learn 7 days in a row', bs >= 7],
    ['M', 'Module cleared', 'Finish every lesson in a module', full],
    ['10', 'Warmed up', 'Finish 10 lessons', n >= 10],
    ['★', 'Sharp', 'Rate 10 answers “Nailed it”', nail >= 10],
    ['20', 'Recall', 'Do 20 flashcard reviews', rv >= 20],
    ['22', 'Halfway', 'Finish 22 lessons', n >= 22],
    ['∞', 'Shipped', 'Finish all 43 lessons', n >= ALL.length],
  ];
}
function bump(){ const d = today(); st.act[d] = (st.act[d] || 0) + 1; lsSet(); if (db) db.doc('activity/' + d).set({n: st.act[d], at: now()}).catch(() => {}); }
function award(delta, label, before){
  toast(`<b>+${delta} XP</b> ${esc(label)}`);
  const a = level(before), b = level();
  if (b.i > a.i) setTimeout(() => { toast(`<b>Level ${b.i+1}</b> You're now ${esc(b.name)}`, true); confetti(160); }, 500);
  chrome();
}

// ---------- spaced review ----------
const BOX = [0, 1, 3, 7, 14, 30];
function dueCards(){
  const t = today();
  return ALL.filter(l => isDone(l.id) && l.s).filter(l => { const r = st.rev[key(l.id)]; return !r || !r.due || r.due <= t; });
}

// ---------- diagrams ----------
const TW = (s, px=12) => Math.ceil(String(s).length * px * 0.56);
const AV = () => { const w = window.innerWidth || 800; const side = w >= 960 ? 272 : 0; const rail = w >= 1280 ? 274 : 0; return Math.max(240, Math.min(w - side - rail - (w < 641 ? 64 : w < 960 ? 90 : 116), 720)); };
const ARW = '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="currentColor"/></marker></defs>';
function parseLane(line){
  let title = ''; const m = line.match(/^\[([^\]]+)\]\s*(.*)$/); if (m){ title = m[1]; line = m[2]; }
  const parts = line.split(/\s+(?:-\(((?:[^()]|\([^()]*\))*)\)-|-)>\s+/);
  const nodes = [], labels = [];
  for (let k = 0; k < parts.length; k++){ if (k % 2 === 0) nodes.push(parts[k].trim()); else labels.push(parts[k] || ''); }
  return {title, nodes: nodes.map(n => ({t: n.replace(/^\*/,''), hot: n.startsWith('*')})), labels};
}
function box(x, y, w, h, n){
  const c = n.hot ? 'var(--accent)' : 'currentColor', f = n.hot ? 'var(--accent-soft)' : 'var(--card)', sh = n.hot ? 'var(--accent)' : 'var(--dg-sh)';
  return `<rect x="${x}" y="${y+3}" width="${w}" height="${h}" rx="9" fill="${sh}"/>` +
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9" fill="${f}" stroke="${c}" stroke-width="1.3"/>` +
    `<text x="${x+w/2}" y="${y+h/2+4}" text-anchor="middle" font-size="12.5" font-weight="600" fill="${n.hot?'var(--accent)':'currentColor'}">${esc(n.t)}</text>`;
}
function flowSVG(lines, force){
  const lanes = lines.map(parseLane), H = 36, out = [];
  let W = 0; const hl = lanes.map(L => { let x = 4; const pos = L.nodes.map((n, k) => { const w = Math.max(64, TW(n.t, 12.5) + 24); const p = {x, w}; x += w; if (k < L.labels.length) x += Math.max(40, TW(L.labels[k], 11) + 16); return p; }); W = Math.max(W, x + 4); return pos; });
  if (force || W <= AV() * 1.15){
    let y = 4;
    lanes.forEach((L, li) => {
      if (L.title){ out.push(`<text x="4" y="${y+12}" font-size="10.5" font-weight="600" fill="var(--muted)" letter-spacing=".12em" font-family="var(--mono)">${esc(L.title.toUpperCase())}</text>`); y += 20; }
      y += L.labels.some(s => s) ? 16 : 0;
      hl[li].forEach((p, k) => { out.push(box(p.x, y, p.w, H, L.nodes[k]));
        if (k < L.labels.length){ const x1 = p.x + p.w + 3, x2 = hl[li][k+1].x - 3, cy = y + H/2;
          out.push(`<line x1="${x1}" y1="${cy}" x2="${x2}" y2="${cy}" stroke="currentColor" stroke-width="1.3" marker-end="url(#ah)"/>`);
          if (L.labels[k]) out.push(`<text x="${(x1+x2)/2}" y="${cy-8}" text-anchor="middle" font-size="11" fill="var(--muted)">${esc(L.labels[k])}</text>`); } });
      y += H + 18;
    });
    return {W, Hh: y - 8, body: out.join('')};
  }
  const nw = Math.max(...lanes.flatMap(L => L.nodes.map(n => Math.max(90, TW(n.t, 12.5) + 24))));
  const lw = Math.max(0, ...lanes.flatMap(L => L.labels.map(s => TW(s, 11))));
  const cx = nw / 2 + 4; W = Math.max(nw + 8, cx + 12 + lw + 4); let y = 4;
  lanes.forEach(L => {
    if (L.title){ out.push(`<text x="${cx}" y="${y+12}" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--muted)" letter-spacing=".12em" font-family="var(--mono)">${esc(L.title.toUpperCase())}</text>`); y += 20; }
    L.nodes.forEach((n, k) => { const w = Math.max(90, TW(n.t, 12.5) + 24); out.push(box(cx - w/2, y, w, H, n)); y += H;
      if (k < L.labels.length){ out.push(`<line x1="${cx}" y1="${y+6}" x2="${cx}" y2="${y+32}" stroke="currentColor" stroke-width="1.3" marker-end="url(#ah)"/>`);
        if (L.labels[k]) out.push(`<text x="${cx+10}" y="${y+23}" font-size="11" fill="var(--muted)">${esc(L.labels[k])}</text>`); y += 36; } });
    y += 22;
  });
  return {W, Hh: y - 16, body: out.join('')};
}
function seqSVG(lines){
  const ps = [], msgs = [];
  lines.forEach(l => { const m = l.match(/^(.+?)\s*(-?)>\s*(.+?)\s*:\s*(.*)$/); if (!m) return; const a = m[1].trim(), b = m[3].trim();
    [a, b].forEach(p => { if (!ps.includes(p)) ps.push(p); }); msgs.push({a, b, t: m[4], dash: m[2] === '-'}); });
  const col = Math.max(96, ...ps.map(p => TW(p, 12.5) + 20), ...msgs.map(m => Math.min(TW(m.t, 11) + 18, 230)));
  const W = col * ps.length, X = p => col * ps.indexOf(p) + col / 2, top = 38, rh = 32, Hh = top + msgs.length * rh + 14;
  const out = [];
  ps.forEach(p => { const x = X(p), w = Math.min(col - 10, TW(p, 12.5) + 24);
    out.push(`<line x1="${x}" y1="${top}" x2="${x}" y2="${Hh-4}" stroke="var(--line2)" stroke-width="1.5" stroke-dasharray="4 4"/>`);
    out.push(box(x - w/2, 2, w, 30, {t: p})); });
  msgs.forEach((m, k) => { const y = top + 18 + k * rh, x1 = X(m.a), x2 = X(m.b);
    if (m.a === m.b){ const last = ps.indexOf(m.a) === ps.length - 1, s = last ? -1 : 1; out.push(`<path d="M${x1} ${y-6}h${22*s}v12h${-19*s}" fill="none" stroke="currentColor" stroke-width="1.3" marker-end="url(#ah)"/><text x="${x1+28*s}" y="${y+4}" text-anchor="${last?'end':'start'}" font-size="11" fill="var(--muted)">${esc(m.t)}</text>`); return; }
    const d = x2 > x1 ? 1 : -1;
    out.push(`<line x1="${x1+d*3}" y1="${y}" x2="${x2-d*3}" y2="${y}" stroke="${m.dash?'var(--muted)':'currentColor'}" stroke-width="1.3" ${m.dash?'stroke-dasharray="5 4"':''} marker-end="url(#ah)"/>`);
    out.push(`<text x="${(x1+x2)/2}" y="${y-6}" text-anchor="middle" font-size="11" fill="var(--muted)">${esc(m.t)}</text>`); });
  return {W, Hh, body: out.join('')};
}
function layersHTML(lines){
  return '<div class="layers">' + lines.map((l, k) => { const [n, note=''] = l.split(/\s+\|\s+/); const hot = n.startsWith('*');
    return (k ? '<div class="lay-arr" aria-hidden="true">↕</div>' : '') + `<div class="lay${hot?' hot':''}"><strong>${inl(n.replace(/^\*/,''))}</strong>${note?`<span>${inl(note)}</span>`:''}</div>`; }).join('') + '</div>';
}
function svgOf(d, cap, wide){ return `<svg viewBox="0 0 ${d.W} ${d.Hh}" role="img" aria-label="${esc(cap)}" style="${wide?`width:${d.W}px;max-width:none`:`width:100%;max-width:${d.W}px`};height:auto;color:var(--ink);font-family:var(--body)">${ARW}${d.body}</svg>`; }
function diagram(kind, cap, lines){
  let inner;
  if (kind === 'layers') inner = layersHTML(lines);
  else { const d = kind === 'seq' ? seqSVG(lines) : flowSVG(lines); const wide = d.W > AV() * 1.25;
    inner = `<div class="dg-scroll">${svgOf(d, cap, wide)}</div>`; }
  if (inner.includes('max-width:none')) inner += '<div class="dg-hint">← scroll sideways to see the whole diagram →</div>';
  return `<figure class="dg">${inner}${cap?`<figcaption>${inl(cap)}</figcaption>`:''}</figure>`;
}

// ---------- notes markup ----------
function render(src){
  const lines = src.split('\n'); let out = '', i = 0;
  while (i < lines.length){
    const L = lines[i];
    if (!L.trim()){ i++; continue; }
    if (L.startsWith('## ')){ out += `<h2>${inl(L.slice(3))}</h2>`; i++; continue; }
    if (L.startsWith('- ')){ out += '<ul>'; while (i < lines.length && lines[i].startsWith('- ')){ out += `<li>${inl(lines[i].slice(2))}</li>`; i++; } out += '</ul>'; continue; }
    if (/^%%(flow|seq|layers)\b/.test(L)){ const mm = L.match(/^%%(\w+)\s*(.*)$/), b = []; i++; while (i < lines.length && lines[i].trim() !== '%%'){ if (lines[i].trim()) b.push(lines[i].trim()); i++; } i++; out += diagram(mm[1], mm[2], b); continue; }
    if (L.startsWith('```')){ const c=[]; i++; while (i < lines.length && !lines[i].startsWith('```')){ c.push(lines[i]); i++; } i++; out += `<pre class="code">${esc(c.join('\n'))}</pre>`; continue; }
    if (/^\d+\. /.test(L)){ out += '<ol>'; while (i < lines.length && /^\d+\. /.test(lines[i])){ out += `<li>${inl(lines[i].replace(/^\d+\. /,''))}</li>`; i++; } out += '</ol>'; continue; }
    if (L.startsWith('> ')){ out += `<div class="take"><div class="lab">Key takeaway</div><p>${inl(L.replace(/^> (Takeaway: )?/,''))}</p></div>`; i++; continue; }
    if (L.startsWith('! ')){ const b = []; while (i < lines.length && lines[i].startsWith('! ')){ b.push(lines[i].slice(2)); i++; }
      const t = b[0].replace(/^Example:\s*/,''); out += `<div class="ex"><div class="lab">Worked example</div><p><strong>${inl(t)}</strong></p>${b.slice(1).map(x => `<p>${inl(x)}</p>`).join('')}</div>`; continue; }
    if (L.startsWith('| ')){ out += '<div class="tbl"><table>'; while (i < lines.length && lines[i].startsWith('| ')){ out += '<tr>' + lines[i].slice(2).replace(/\s*\|\s*$/,'').split(' | ').map(c => `<td>${inl(c)}</td>`).join('') + '</tr>'; i++; } out += '</table></div>'; continue; }
    const p = []; while (i < lines.length && lines[i].trim() && !/^(## |- |> |! |\| |\d+\. |```|%%)/.test(lines[i])){ p.push(lines[i]); i++; }
    if (!p.length){ out += `<p>${inl(L)}</p>`; i++; continue; }
    out += `<p>${inl(p.join(' '))}</p>`;
  }
  return out;
}

// ---------- chrome: top bar, sidebar ----------
let cur = {page: 'home', mod: null, k: null};
function crumbs(parts){ $('#crumbs').innerHTML = parts.map((p, i) => (i ? '<span class="sep">/</span>' : '') + `<a href="${p[1]}">${esc(p[0])}</a>`).join(''); }
function chrome(){
  const c = dayCounts(), s = streak(c), L = level();
  const cs = $('#chipS'); cs.className = 'chip st' + (s ? '' : ' off'); cs.innerHTML = `${FLAME}${s}<span class="xl">&nbsp;day${s===1?'':'s'}</span>`;
  $('#chipX').innerHTML = `<span class="lv">Lv${L.i+1}</span><span class="xl">${L.x} XP</span>`;
  side();
}
function side(){
  const due = dueCards().length, m = cur.mod;
  $('#side').innerHTML = `
    <a class="nv" href="#" ${cur.page==='home'?'aria-current="page"':''}><svg width="16" height="16" viewBox="0 0 16 16"><rect x="2" y="2" width="5" height="5" rx="1.5" fill="currentColor"/><rect x="9" y="2" width="5" height="5" rx="1.5" fill="currentColor" opacity=".45"/><rect x="2" y="9" width="5" height="5" rx="1.5" fill="currentColor" opacity=".45"/><rect x="9" y="9" width="5" height="5" rx="1.5" fill="currentColor"/></svg>Overview</a>
    <a class="nv" href="#review" ${cur.page==='review'?'aria-current="page"':''}><svg width="16" height="16" viewBox="0 0 16 16"><rect x="3" y="4" width="10" height="9" rx="2" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M5 2h6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>Review deck${due ? `<span class="cnt">${due}</span>` : ''}</a>
    ${PARTS.map((P, pi) => `<div class="part lab"><span>Part ${pi+1} · ${P.t}</span></div>` +
      DATA.filter(x => +x.n >= P.a && +x.n <= P.b).map(x => { const d = modDone(x), t = x.lessons.length, on = x.n === m;
        return `<a class="mr${d===t?' full':''}" href="#m${x.n}" ${on?'aria-current="true"':''}><span class="n">${x.n}</span><span>${esc(x.name)}</span><span class="c">${d}/${t}</span></a>` +
          (on ? `<div class="ls">${x.lessons.map(l => `<a class="li" href="#l${key(l.id)}" ${key(l.id)===cur.k?'aria-current="page"':''}><span class="sd${isDone(l.id)?' on':''}"></span><span>${esc(l.t)}</span></a>`).join('')}</div>` : ''); }).join('')).join('')}`;
}
function closeSide(){ $('#side').classList.remove('open'); $('#scrim').classList.remove('on'); }
$('#menu').onclick = () => { $('#side').classList.add('open'); $('#scrim').classList.add('on'); };
$('#scrim').onclick = closeSide;
$('#side').addEventListener('click', e => { if (e.target.closest('a')) closeSide(); });

// ---------- pages ----------
const nextUp = () => ALL.find(l => !isDone(l.id));
const segs = (m, curK) => `<div class="seg" aria-hidden="true">${m.lessons.map(l => `<i class="${isDone(l.id)?'on':key(l.id)===curK?'cur':''}"></i>`).join('')}</div>`;
function ringSVG(v, max, size = 54){ const r = size/2 - 5, C = 2 * Math.PI * r, p = Math.min(1, v / max);
  return `<svg class="ring" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" aria-hidden="true"><circle cx="${size/2}" cy="${size/2}" r="${r}" fill="none" stroke="var(--line)" stroke-width="6"/><circle cx="${size/2}" cy="${size/2}" r="${r}" fill="none" stroke="${p>=1?'var(--go)':'var(--accent)'}" stroke-width="6" stroke-linecap="${p>0?"round":"butt"}" stroke-dasharray="${C*p} ${C}" transform="rotate(-90 ${size/2} ${size/2})"/></svg>`; }

const G = s => `<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${s}</svg>`;
const GLYPH = {
  '01': G('<circle cx="9" cy="38" r="3.5" fill="currentColor"/><circle cx="39" cy="10" r="3.5" fill="currentColor"/><path d="M12 36c10-4 4-14 13-16s7-6 11-8" stroke-dasharray="3 4"/>'),
  '02': G('<path d="M8 40V8M8 40h32"/><circle cx="18" cy="28" r="2.5" fill="currentColor"/><circle cx="22" cy="31" r="2.5" fill="currentColor"/><circle cx="31" cy="15" r="2.5" fill="currentColor"/><circle cx="35" cy="19" r="2.5" fill="currentColor"/><path d="M8 40L33 17" opacity=".6"/>'),
  '03': G('<path d="M8 12h28M8 20h32M8 28h22"/><path d="M8 36h12"/><path d="M24 31v10" stroke-width="2.6"/>'),
  '04': G('<rect x="9" y="8" width="30" height="8" rx="2"/><rect x="9" y="20" width="30" height="8" rx="2"/><rect x="9" y="32" width="30" height="8" rx="2" fill="currentColor"/>'),
  '05': G('<circle cx="10" cy="36" r="3"/><circle cx="24" cy="36" r="3"/><circle cx="38" cy="36" r="3" fill="currentColor"/><path d="M10 33c0-14 28-14 28 0M24 33c0-8 14-8 14 0"/><path d="M10 33c0-22 28-22 28 0" opacity=".5"/>'),
  '06': G('<path d="M27 6L12 27h11l-3 15 16-22H25z" fill="currentColor" fill-opacity=".25"/>'),
  '07': G('<rect x="6" y="12" width="20" height="24" rx="3"/><rect x="31" y="22" width="11" height="14" rx="2.5" fill="currentColor"/><path d="M26 29h5" stroke-dasharray="2 3"/>'),
  '08': G('<circle cx="24" cy="8" r="3" fill="currentColor"/><path d="M24 11v7M24 18l-12 10M24 18l12 10M12 28v8M36 28v8"/><circle cx="12" cy="39" r="3"/><circle cx="36" cy="39" r="3" fill="currentColor"/><path d="M21 25l-3 3m0-3l3 3" opacity=".7"/>'),
  '09': G('<path d="M36 18a14 14 0 1 0 2 10"/><path d="M38 9v10h-10"/><circle cx="24" cy="24" r="4" fill="currentColor"/>'),
};
const COV = [['#5A4FCF','#fff'],['#1F7A72','#fff'],['#E9B872','#272727'],['#C2577A','#fff'],['#A3B8E8','#272727'],['#2B2B3A','#fff']];
function home(){
  crumbs([]);
  const nx = nextUp(), n = doneCount(), nm = nx && DATA.find(m => m.n === nx.mod), c = dayCounts(), s = streak(c), L = level(), tl = todayLessons(), due = dueCards().length;
  const hr = new Date().getHours(), greet = hr < 12 ? 'Good morning' : hr < 18 ? 'Good afternoon' : 'Good evening';
  const wk = Array.from({length: 7}, (_, i) => addDays(today(), i - 6));
  const TOK = ['The', 'model', 'reads', 'each', 'token'];
  const AW = [[1],[.35,.65],[.15,.55,.3],[.1,.2,.5,.2],[.05,.3,.15,.2,.3]];
  const mini = `<div class="att">${TOK.map((t, r) => `<span class="tk">${t}</span>` + TOK.map((_, c) => c > r ? '<i class="m"></i>' : `<i style="--w:${Math.min(1, AW[r][c] * 1.6 + .08).toFixed(2)}"></i>`).join('')).join('')}</div>`;
  const last = ALL.filter(l => isDone(l.id)).slice(-1)[0];
  const end = new Date(today() + 'T12:00:00'), dow = (end.getDay() + 6) % 7, start = addDays(today(), -(14 * 7 + dow));
  const cells = Array.from({length: 15 * 7}, (_, i) => { const d = addDays(start, i), v = c[d] || 0; return `<i class="${d > today() ? 'fut' : v >= 4 ? 'l4' : v === 3 ? 'l3' : v === 2 ? 'l2' : v ? 'l1' : ''}" title="${d}: ${v}"></i>`; }).join('');
  const activeDays = Object.keys(c).filter(d => d >= start).length, B = badges();
  $('#view').innerHTML = `
  <section class="hero panel">
    <div class="lab">${greet} · ${DATA.length} modules · ${ALL.length} lessons</div>
    <h1>How modern AI systems work, <span class="px">one lesson at a time.</span></h1>
    <div class="float f1" aria-hidden="true"><div class="lab">Fig · causal attention</div>${mini}</div>
    <div class="hrow">
      ${nx ? `<div class="note"><div class="lab">${n ? 'Reading note' : 'Start here'} · ${nx.id} · ${mins(nx)} min</div><h3>${esc(nx.t)}</h3><p>${inl(nx.n[0])}</p>
        <a class="btn" href="#l${key(nx.id)}">${n ? 'Continue' : 'Start learning'} <span class="xp">+10 XP</span></a></div>`
        : `<div class="note"><div class="lab">Course complete</div><h3>All ${ALL.length} lessons done.</h3><p>Keep it fresh with the review deck.</p><a class="btn" href="#review">Open review deck</a></div>`}
      <div><p class="hbody">Each lesson is a short read with <b>diagrams and worked examples</b>. You answer a question, <b>compare it with a model answer</b>, and the cards come back for review until they stick.</p>
        <div class="hlinks"><a class="btn ghost" href="#review">Review deck${due ? ` · ${due} due` : ''}</a>${nx ? `<a class="btn ghost" href="#m${nx.mod}">Module ${nx.mod} · ${esc(nm.name)}</a>` : ''}</div></div>
    </div>
    <div class="strip">
      <div><div class="lab">Streak</div><div class="v">${s}<small>day${s===1?'':'s'}</small></div><div class="seg">${wk.map(d => `<i class="${c[d]?'fl':''}"></i>`).join('')}</div></div>
      <div><div class="lab">Today</div><div class="v">${Math.min(tl, GOAL)}<small>/ ${GOAL} lessons</small></div><div class="seg">${Array.from({length: GOAL}, (_, i) => `<i class="${i < tl ? 'on' : ''}"></i>`).join('')}</div></div>
      <div><div class="lab">Level ${L.i+1} · ${esc(L.name)}</div><div class="v">${L.x}<small>XP</small></div><div class="bar"><i style="width:${L.pct*100}%"></i></div></div>
      <div><div class="lab">Course</div><div class="v">${n}<small>/ ${ALL.length}</small></div><div class="bar"><i style="width:${n/ALL.length*100}%;background:#7FD1A8"></i></div></div>
    </div>
  </section>

  <div class="week">
    <div class="card"><div style="display:flex;justify-content:space-between;gap:10px"><div class="lab">Activity · 15 weeks</div><div class="lab">${activeDays} active day${activeDays===1?'':'s'} · best streak ${bestStreak(c)}</div></div>
      <div class="hm">${cells}</div>
      <div class="legend">Less <i style="background:var(--panel)"></i><i style="background:color-mix(in srgb,var(--accent) 30%,var(--card))"></i><i style="background:color-mix(in srgb,var(--accent) 55%,var(--card))"></i><i style="background:color-mix(in srgb,var(--accent) 80%,var(--card))"></i><i style="background:var(--accent)"></i> More</div>
      ${last ? `<div style="margin-top:12px;font-size:13.5px;color:var(--muted)">Last finished: <a href="#l${key(last.id)}" style="color:var(--ink)">${esc(last.t)}</a></div>` : ''}</div>
    <div class="card"><div style="display:flex;justify-content:space-between"><div class="lab">Badges</div><div class="lab">${B.filter(b => b[3]).length} / ${B.length}</div></div>
      <div class="stamps">${B.map(b => `<div class="stp${b[3]?' on':''}" title="${esc(b[2])}"><span class="ic">${b[0]}</span><span><b>${b[1]}</b><span>${b[2]}</span></span></div>`).join('')}</div></div>
  </div>

  <div class="sh"><h2>Course map</h2><span class="lab">${n} / ${ALL.length} lessons</span></div>
  ${PARTS.map((P, pi) => `<div class="parth"><span class="lab">Part ${pi+1}</span><h3>${P.t}</h3></div><div class="mods">` +
    DATA.filter(x => +x.n >= P.a && +x.n <= P.b).map(m => { const d = modDone(m), t = m.lessons.length, [bg, fg] = COV[(+m.n - 1) % COV.length];
      return `<a class="card mcard" href="#m${m.n}"><div class="cover" style="--c:${bg};--ci:${fg}"><div class="tp"><span class="cn">${m.n}</span>${d === t ? `<span class="ck">${CHECK}</span>` : ''}</div><span class="tg">${d === t ? 'Complete' : d ? `${d} of ${t} done` : `${t} lessons`}</span><span class="gl" aria-hidden="true">${GLYPH[m.n] || ''}</span></div>
        <h4>${esc(m.name)}</h4>${segs(m)}<div class="ft"><span>${d}/${t}</span><span>~${m.lessons.reduce((a, l) => a + mins(l), 0)} min</span></div></a>`; }).join('') + '</div>').join('')}
  <p class="foot-note">Notes are written in my own words from the linked sources. Lesson titles follow the fanout.sh AI engineering course. If a source link is broken or paywalled, press “report” on the lesson page.</p>`;
}

function modulePage(n){
  const m = DATA.find(x => x.n === n); if (!m) return home();
  crumbs([[`Module ${m.n}`, `#m${m.n}`]]);
  const c = modDone(m), t = m.lessons.length, first = m.lessons.find(l => !isDone(l.id)) || m.lessons[0];
  const i = DATA.indexOf(m), prev = DATA[i-1], next = DATA[i+1], total = m.lessons.reduce((a, l) => a + mins(l), 0);
  const curL = m.lessons.find(l => !isDone(l.id));
  $('#view').innerHTML = `
  <section class="panel mhead"><div>
    <div class="lab">Module ${m.n} of ${DATA.length} · Part ${PARTS.findIndex(P => +m.n >= P.a && +m.n <= P.b) + 1}</div>
    <h1>${esc(m.name)}</h1>
    <div class="meta"><span class="pill">${t} lessons</span><span class="pill">~${total} min</span><span class="pill ${c===t?'ok':c?'ip':''}">${c}/${t} done</span><span class="pill k">+${t*10} XP</span></div>
    <div style="margin-top:20px;display:flex;gap:16px;align-items:center;flex-wrap:wrap"><a class="btn" href="#l${key(first.id)}">${c ? (c === t ? 'Review module' : 'Continue') : 'Start module'} →</a><div style="flex:1;min-width:180px">${segs(m, curL && key(curL.id))}</div></div>
  </div><div class="big" aria-hidden="true">${m.n}</div></section>
  <div class="path">${m.lessons.map((l, j) => { const d = isDone(l.id), cu = curL === l, dx = Math.round(Math.sin(j * 0.9) * 70);
    return `<a class="pnode${d?' done':''}${cu?' cur':''}" href="#l${key(l.id)}" style="--dx:${dx}px"><span class="o">${d ? CHECK : j + 1}</span><span class="t"><span>${l.id} · ${mins(l)} min${cu ? ' · up next' : ''}</span><b>${esc(l.t)}</b></span></a>`; }).join('')}</div>
  <div class="mnav">${prev ? `<a class="btn ghost" href="#m${prev.n}">← ${esc(prev.name)}</a>` : '<span></span>'}${next ? `<a class="btn ghost" href="#m${next.n}">${esc(next.name)} →</a>` : ''}</div>`;
}

function lessonPage(k){
  const l = ALL.find(x => key(x.id) === k); if (!l) return home();
  const m = DATA.find(x => x.n === l.mod), gi = ALL.indexOf(l), prev = ALL[gi-1], next = ALL[gi+1];
  crumbs([[`${m.n} · ${m.name}`, `#m${m.n}`], [l.t, `#l${k}`]]);
  const d = isDone(l.id), g = st.grades[k]?.g || 0, rv = !!st.reveal[k] || !!g;
  const tags = [...l.k.map(t => `<span class="pill k">${esc(t)}</span>`), ...l.x.map(t => `<span class="pill">${esc(t)}</span>`), l.ak ? `<span class="pill a">Asked at ${esc(l.ak)}</span>` : ''].join('');
  const body = l.a ? render(l.a) : `<div class="wip">Full notes for this lesson are still being written. Here are the key ideas for now.</div><ul>${l.n.map(x => `<li>${inl(x)}</li>`).join('')}</ul>`;
  const srcs = l.r.map((r, i) => { const fk = k + '_' + i, on = !!st.flags[fk], h = host(r[1]);
    return `<li><span class="fav" aria-hidden="true">${esc((h[0] || '?').toUpperCase())}</span><span class="tx"><a href="${esc(r[1])}" target="_blank" rel="noopener">${esc(r[0])}</a><span class="n">${esc(r[2])}</span><span class="h">${esc(h)}</span></span><button type="button" class="rep${on?' on':''}" data-flag="${fk}" data-url="${esc(r[1])}" aria-pressed="${on}">${on?'reported':'report'}</button></li>`; }).join('');
  const gbtn = (v, t, s) => `<button type="button" class="gbtn" data-g="${v}" aria-pressed="${g===v}">${t}<small>+${GB[v]} XP</small></button>`;
  $('#view').innerHTML = `
  <div class="lgrid">
    <article>
      <header class="panel lhead">
        <a class="back" href="#m${m.n}">← Module ${m.n} · ${esc(m.name)}</a>
        <div class="idrow"><span class="lid">${l.id}</span><span class="lab">Lesson ${l.idx+1} of ${m.lessons.length}</span>${d ? '<span class="pill ok">✓ Completed</span>' : ''}</div>
        <h1>${esc(l.t)}</h1>
        <div class="meta"><span class="pill">${mins(l)} min read</span>${tags}</div>
        ${segs(m, k)}
      </header>
      <div class="prose" id="prose">${body}</div>

      <section class="card blk" id="check"><div class="lab">Check yourself</div>
        <p class="q">${esc(l.q)}</p>
        <textarea id="ans-${k}" data-note="${k}" aria-label="Your answer" placeholder="Write your answer first. It saves automatically.">${esc(st.notes[k] || '')}</textarea>
        <div class="arow"><span class="saved" id="saved-${k}"></span>${l.s ? (rv ? `<button type="button" class="btn ghost" id="reveal" style="min-height:38px;font-size:14px">Hide model answer</button>` : `<button type="button" class="btn" id="reveal">Check against model answer</button>`) : ''}</div>
        ${l.s && rv ? `<div class="sample"><div class="lab">Model answer</div><p>${inl(l.s)}</p></div>
          <div class="grade"><div class="lab">How did your answer compare?</div><div class="gb">${gbtn(1, 'Missed it')}${gbtn(2, 'Partly')}${gbtn(3, 'Nailed it')}</div></div>` : ''}
      </section>

      <section class="card blk"><div class="lab">Go deeper</div><h3>Sources these notes are based on</h3><p style="color:var(--muted);margin:0 0 16px;font-size:14.5px">Original lesson on <a href="${esc(l.f)}" target="_blank" rel="noopener" style="color:var(--muted)">fanout.sh</a>.</p><ul class="srcs">${srcs}</ul></section>

      <div class="done-zone">
        ${d ? `<button type="button" class="btn go" id="undo">${CHECK} Completed · undo</button>`
            : `<button type="button" class="btn" id="complete">Mark complete <span class="xp">+10 XP</span></button>`}
        <p>Shortcuts: C complete · J next · K previous · / search</p>
      </div>
      <nav class="pn" aria-label="Lesson navigation">
        ${prev ? `<a class="card" href="#l${key(prev.id)}"><span class="lab">← Previous · ${prev.id}</span><b>${esc(prev.t)}</b></a>` : ''}
        ${next ? `<a class="card nx${d?' hl':''}" href="#l${key(next.id)}"><span class="lab">Next · ${next.id} →</span><b>${esc(next.t)}</b></a>` : ''}
      </nav>
    </article>
    <aside class="rail">
      <div class="card"><div class="lab" style="margin-bottom:10px">On this page</div><div class="toc" id="toc"></div></div>
      <div class="card"><div class="lab" style="margin-bottom:10px">Module progress</div><div style="font:600 26px var(--body);letter-spacing:-.03em;margin-bottom:8px">${modDone(m)}<span style="font:500 12px var(--mono);color:var(--muted)"> / ${m.lessons.length}</span></div>${segs(m, k)}</div>
    </aside>
  </div>`;
  const hs = [...document.querySelectorAll('#prose h2')]; hs.forEach((h, i) => h.id = 's' + (i + 1));
  $('#toc').innerHTML = hs.map(h => `<a href="#l${k}" data-to="${h.id}">${esc(h.textContent)}</a>`).join('') + `<a href="#l${k}" data-to="check">Check yourself</a>`;
  const c = $('#complete'); if (c) c.onclick = () => complete(l, next);
  const u = $('#undo'); if (u) u.onclick = () => { setDone(k, false); rerender(); };
  const r = $('#reveal'); if (r) r.onclick = () => { st.reveal[k] = !rv; if (rv && g) st.reveal[k] = false; rerender(); if (!rv) setTimeout(() => $('.sample')?.scrollIntoView({behavior: RM ? 'auto' : 'smooth', block: 'center'}), 30); };
  document.querySelectorAll('.gbtn').forEach(b => b.onclick = () => grade(k, +b.dataset.g));
}
function complete(l, next){
  const k = key(l.id), m = DATA.find(x => x.n === l.mod), before = xp();
  setDone(k, true); confetti(90);
  award(10, 'Lesson complete', before);
  if (modDone(m) === m.lessons.length) setTimeout(() => { toast(`<b>Module ${m.n}</b> cleared: ${esc(m.name)}`, true); confetti(260); }, 450);
  else if (todayLessons() === GOAL) setTimeout(() => toast('<b>Goal</b> Daily goal reached'), 600);
  rerender();
}
function grade(k, v){
  const before = xp(), old = st.grades[k]?.g || 0; if (old === v) return;
  st.grades[k] = {g: v, at: now()}; lsSet(); bump();
  if (db) db.doc('grades/' + k).set({g: v, at: st.grades[k].at}).catch(() => {});
  const delta = xp() - before;
  if (delta > 0) award(delta, v === 3 ? 'Nailed it' : v === 2 ? 'Good effort' : 'Honest grading', before); else { toast(v === 1 ? 'Saved. Re-read the sections above.' : 'Grade updated'); chrome(); }
  if (v === 3 && !RM) confetti(50);
  rerender();
}

// ---------- review deck ----------
let rq = null, rflip = false, rsess = 0;
function reviewPage(){
  crumbs([['Review deck', '#review']]);
  if (!rq) rq = dueCards().sort(() => Math.random() - .5).slice(0, 20).map(l => key(l.id));
  const doneN = ALL.filter(l => isDone(l.id) && l.s).length;
  if (!rq.length){
    $('#view').innerHTML = `<div class="rv"><div class="rvtop"><h1 style="margin:0;font-size:34px;letter-spacing:-.045em">Review deck</h1></div>
      <div class="card empty"><div class="big">${doneN ? '✓' : '0'}</div><h3 style="font-size:22px;letter-spacing:-.03em;margin:14px 0 6px">${doneN ? (rsess ? `Session done: ${rsess} card${rsess===1?'':'s'}` : 'Nothing due right now') : 'No cards yet'}</h3>
      <p style="color:var(--muted);margin:0 0 20px">${doneN ? 'Cards come back on a schedule: 1, 3, 7, 14, then 30 days after you get them right.' : 'Each lesson you complete adds a flashcard with its question and model answer.'}</p>
      <a class="btn" href="${nextUp() ? '#l' + key(nextUp().id) : '#'}">${nextUp() ? 'Go to next lesson' : 'Back to overview'} →</a></div></div>`;
    return;
  }
  const l = ALL.find(x => key(x.id) === rq[0]), mine = st.notes[rq[0]];
  $('#view').innerHTML = `<div class="rv">
    <div class="rvtop"><h1 style="margin:0;font-size:34px;letter-spacing:-.045em">Review deck</h1><span class="lab">${rq.length} left · +5 XP each</span></div>
    <div class="flip${rflip?' on':''}" id="flip" role="button" tabindex="0" aria-label="Flip card"><div class="in">
      <div class="face fr"><div class="lab">Module ${l.mod} · ${l.id} · ${esc(l.t)}</div><p class="q">${esc(l.q)}</p><div class="hint">Answer in your head, then tap the card or press space to flip</div></div>
      <div class="face bk"><div class="lab">Model answer</div><p style="margin:0;font-size:16px;line-height:1.6">${inl(l.s)}</p>${mine ? `<div class="mine"><div class="lab" style="margin-bottom:6px">Your saved answer</div>${esc(mine.length > 400 ? mine.slice(0, 397) + '…' : mine)}</div>` : ''}<div class="hint" style="color:#8A8A93"><a href="#l${key(l.id)}" style="color:#BDBDC5">Open lesson →</a></div></div>
    </div></div>
    <div class="rvbtns" style="visibility:${rflip?'visible':'hidden'}">
      <button type="button" class="gbtn" data-r="1" aria-pressed="true" data-g="1">Again<small>soon</small></button>
      <button type="button" class="gbtn" data-r="2" aria-pressed="true" data-g="2">Hard<small>tomorrow</small></button>
      <button type="button" class="gbtn" data-r="3" aria-pressed="true" data-g="3">Got it<small>later</small></button>
    </div></div>`;
  const f = $('#flip'); f.onclick = e => { if (e.target.closest('a')) return; rflip = !rflip; f.classList.toggle('on', rflip); $('.rvbtns').style.visibility = rflip ? 'visible' : 'hidden'; };
  document.querySelectorAll('[data-r]').forEach(b => b.onclick = () => rate(+b.dataset.r));
}
function rate(v){
  const k = rq.shift(), before = xp(), r = st.rev[k] || {n: 0, box: 0};
  r.n = (r.n || 0) + 1; r.at = now();
  if (v === 1){ r.box = 0; r.due = today(); rq.push(k); }
  else if (v === 2){ r.box = Math.max(1, r.box || 0); r.due = addDays(today(), 1); }
  else { r.box = Math.min(5, (r.box || 0) + 1); r.due = addDays(today(), BOX[r.box]); }
  st.rev[k] = r; lsSet(); bump(); rsess++;
  if (db) db.doc('reviews/' + k).set(r).catch(() => {});
  rflip = false; award(5, v === 3 ? 'Recalled' : 'Reviewed', before); reviewPage();
  if (!rq.length) confetti(120);
}

// ---------- search palette ----------
let palSel = 0, palList = [];
function openPal(){ $('#pal').classList.add('on'); const i = $('#palin'); i.value = ''; palRender(); setTimeout(() => i.focus(), 10); }
function closePal(){ $('#pal').classList.remove('on'); }
function palRender(){
  const q = $('#palin').value.trim().toLowerCase();
  if (!q) palList = (nextUp() ? [nextUp()] : []).concat(ALL.filter(l => !isDone(l.id)).slice(1, 8));
  else { const t = [], b = []; ALL.forEach(l => { const ti = (l.t + ' ' + l.id).toLowerCase().includes(q); if (ti) t.push(l); else if ((l.a || '').toLowerCase().includes(q)) b.push(l); }); palList = t.concat(b).slice(0, 14); }
  palSel = Math.min(palSel, Math.max(0, palList.length - 1));
  $('#palres').innerHTML = palList.length ? palList.map((l, i) => { const m = DATA.find(x => x.n === l.mod);
    return `<a href="#l${key(l.id)}" class="${i===palSel?'on':''}" data-i="${i}"><span class="id">${l.id}</span><span class="tt"><b>${esc(l.t)}</b><span>${esc(m.name)}${isDone(l.id) ? ' · done' : ''}</span></span></a>`; }).join('')
    : '<div style="padding:18px;color:var(--muted)">No lessons match.</div>';
}
$('#palin').addEventListener('input', () => { palSel = 0; palRender(); });
$('#palin').addEventListener('keydown', e => {
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp'){ e.preventDefault(); palSel = (palSel + (e.key === 'ArrowDown' ? 1 : -1) + palList.length) % Math.max(1, palList.length); palRender(); document.querySelector('#palres a.on')?.scrollIntoView({block: 'nearest'}); }
  else if (e.key === 'Enter' && palList[palSel]){ location.hash = '#l' + key(palList[palSel].id); closePal(); }
  else if (e.key === 'Escape') closePal();
});
$('#pal').addEventListener('click', e => { if (e.target.id === 'pal' || e.target.closest('#palres a')) closePal(); });
$('#sbtn').onclick = openPal;
document.addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k'){ e.preventDefault(); $('#pal').classList.contains('on') ? closePal() : openPal(); return; }
  const tg = e.target.tagName; if (tg === 'TEXTAREA' || tg === 'INPUT' || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === '/'){ e.preventDefault(); openPal(); return; }
  if (e.key === 'Escape'){ closeSide(); closePal(); }
  if (cur.page === 'lesson'){
    const gi = ALL.findIndex(x => key(x.id) === cur.k);
    if (e.key === 'j' && ALL[gi+1]) location.hash = '#l' + key(ALL[gi+1].id);
    if (e.key === 'k' && ALL[gi-1]) location.hash = '#l' + key(ALL[gi-1].id);
    if (e.key === 'c') ($('#complete') || {click(){}}).click();
  }
  if (cur.page === 'review' && rq && rq.length){
    if (e.key === ' '){ e.preventDefault(); $('#flip')?.click(); }
    if (rflip && ['1','2','3'].includes(e.key)) rate(+e.key);
  }
});

// ---------- feedback: toasts, confetti ----------
function toast(html, big){ const t = document.createElement('div'); t.className = 'toast' + (big ? ' big' : ''); t.innerHTML = html; $('#toasts').appendChild(t); setTimeout(() => t.remove(), big ? 3200 : 2200); }
function confetti(n){
  if (RM) return; const cv = $('#cf'), x = cv.getContext('2d'); cv.width = innerWidth; cv.height = innerHeight;
  const cols = ['#5A4FCF', '#1F7A72', '#E9B872', '#C2577A', '#A3B8E8', '#2B2B3A'];
  const ps = Array.from({length: n}, () => ({x: innerWidth / 2 + (Math.random() - .5) * 160, y: innerHeight * .62, vx: (Math.random() - .5) * 14, vy: -Math.random() * 15 - 6, s: Math.random() * 6 + 4, r: Math.random() * 6, vr: (Math.random() - .5) * .4, c: cols[Math.floor(Math.random() * cols.length)]}));
  let f = 0; (function tick(){ x.clearRect(0, 0, cv.width, cv.height); ps.forEach(p => { p.x += p.vx; p.y += p.vy; p.vy += .42; p.vx *= .985; p.r += p.vr; x.save(); x.translate(p.x, p.y); x.rotate(p.r); x.fillStyle = p.c; x.fillRect(-p.s/2, -p.s/4, p.s, p.s/2); x.restore(); });
    if (++f < 130) requestAnimationFrame(tick); else x.clearRect(0, 0, cv.width, cv.height); })();
}

// ---------- routing ----------
function route(){
  const h = location.hash.slice(1);
  if (/^m\d\d$/.test(h)){ cur = {page: 'module', mod: h.slice(1), k: null}; modulePage(h.slice(1)); }
  else if (/^l\d\d-\d\d$/.test(h)){ const k = h.slice(1); cur = {page: 'lesson', mod: k.slice(0, 2), k}; lessonPage(k); }
  else if (h === 'review'){ cur = {page: 'review', mod: null, k: null}; reviewPage(); }
  else { cur = {page: 'home', mod: null, k: null}; home(); }
  chrome(); onScroll();
}
function rerender(){ const y = scrollY; route(); scrollTo(0, y); }
function onScroll(){
  const rb = $('#readbar');
  if (cur.page !== 'lesson'){ rb.style.width = '0'; return; }
  const H = document.documentElement.scrollHeight - innerHeight; rb.style.width = (H > 0 ? Math.min(1, scrollY / H) * 100 : 0) + '%';
  const as = [...document.querySelectorAll('#toc a')]; let on = null;
  as.forEach(a => { const el = document.getElementById(a.dataset.to); if (el && el.getBoundingClientRect().top < innerHeight * 0.3) on = a; });
  as.forEach(a => a.classList.toggle('on', a === on));
}
addEventListener('scroll', onScroll, {passive: true});
document.addEventListener('click', e => { const a = e.target.closest('#toc a'); if (!a) return; e.preventDefault(); document.getElementById(a.dataset.to)?.scrollIntoView({behavior: RM ? 'auto' : 'smooth'}); });
let lastHash = location.hash;
const bucket = () => { const w = innerWidth || 800; return w < 641 ? 0 : w < 960 ? 1 : w < 1280 ? 2 : 3; };
let _b = bucket(); addEventListener('resize', () => { const b = bucket(); if (b !== _b){ _b = b; if (cur.page !== 'review') rerender(); } });
addEventListener('hashchange', () => { if (location.hash !== '#review') { rq = null; rflip = false; rsess = 0; } route(); if (location.hash !== lastHash) scrollTo(0, 0); lastHash = location.hash; });

// ---------- writes ----------
const timers = {};
async function setDone(k, v){
  st.done[k] = v; if (v) st.at[k] = now(); else delete st.at[k]; lsSet();
  if (!db) return;
  try { v ? await db.doc('progress/' + k).set({done: true, at: st.at[k]}) : await db.doc('progress/' + k).delete(); } catch(e){}
}
document.addEventListener('input', e => {
  const k = e.target.dataset.note; if (!k) return;
  st.notes[k] = e.target.value; lsSet(); clearTimeout(timers[k]);
  const el0 = document.getElementById('saved-' + k); if (el0) el0.textContent = 'Saving…';
  timers[k] = setTimeout(async () => {
    const el = document.getElementById('saved-' + k);
    if (!db){ if (el) el.textContent = 'Saved in this browser'; return; }
    try { const t = st.notes[k]; t.trim() ? await db.doc('notes/' + k).set({text: t, at: now()}) : await db.doc('notes/' + k).delete(); if (el) el.textContent = 'Saved'; }
    catch(err){ if (el) el.textContent = 'Saved in this browser only'; }
  }, 800);
});
document.addEventListener('click', async e => {
  const b = e.target.closest('[data-flag]'); if (!b) return;
  const fk = b.dataset.flag, v = !st.flags[fk]; st.flags[fk] = v; lsSet();
  b.classList.toggle('on', v); b.textContent = v ? 'reported' : 'report'; b.setAttribute('aria-pressed', v);
  if (db) try { v ? await db.doc('flags/' + fk).set({url: b.dataset.url, at: now()}) : await db.doc('flags/' + fk).delete(); } catch(err){}
});

route();

// ---------- shared persistence ----------
(async () => {
  try { db = window.claude && await window.claude.use('db'); } catch(e){ db = null; }
  if (!db) return;
  const sync = () => { if (document.activeElement?.dataset?.note || cur.page === 'review' || $('#pal').classList.contains('on')) chrome(); else rerender(); };
  let first = true;
  db.collection('progress').onSnapshot(snap => {
    const d = {}, at = {}; snap.docs.forEach(s => { const v = s.data(); if (v?.done){ d[s.id] = true; if (v.at) at[s.id] = v.at; } });
    if (first){ Object.keys(st.done).forEach(k => { if (st.done[k] && !d[k]){ d[k] = true; at[k] = st.at[k] || now(); db.doc('progress/' + k).set({done: true, at: at[k]}).catch(() => {}); } }); first = false; }
    const changed = JSON.stringify(d) !== JSON.stringify(Object.fromEntries(Object.entries(st.done).filter(([, v]) => v)));
    st.done = d; st.at = at; lsSet(); changed ? sync() : chrome();
  }, () => {});
  db.collection('notes').onSnapshot(snap => {
    snap.docs.forEach(s => { const t = s.data()?.text; const el = document.getElementById('ans-' + s.id); if (t != null && document.activeElement !== el){ st.notes[s.id] = t; if (el) el.value = t; } });
    lsSet();
  }, () => {});
  db.collection('flags').onSnapshot(snap => {
    const f = {}; snap.docs.forEach(s => f[s.id] = true); st.flags = f; lsSet();
    document.querySelectorAll('[data-flag]').forEach(b => { const on = !!f[b.dataset.flag]; b.classList.toggle('on', on); b.textContent = on ? 'reported' : 'report'; });
  }, () => {});
  const coll = (name, field, map) => db.collection(name).onSnapshot(snap => {
    const o = {}; snap.docs.forEach(s => { const v = s.data(); if (v) o[s.id] = map(v); });
    const ch = JSON.stringify(o) !== JSON.stringify(st[field]); st[field] = o; lsSet(); if (ch) sync();
  }, () => {});
  coll('grades', 'grades', v => ({g: v.g, at: v.at}));
  coll('reviews', 'rev', v => ({n: v.n, box: v.box, due: v.due, at: v.at}));
  coll('activity', 'act', v => v.n || 0);
})();
