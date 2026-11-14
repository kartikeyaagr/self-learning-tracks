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
function box(x, y, w, h, n, s){
  const c = n.hot ? 'var(--accent)' : 'currentColor', f = n.hot ? 'var(--accent-soft)' : 'var(--card)', sh = n.hot ? 'var(--accent)' : 'var(--dg-sh)';
  return `<g class="${s == null ? 'pp' : 'nd'}"${s == null ? '' : ` data-s="${s}" data-t="${esc(n.t)}"`}><rect x="${x}" y="${y+3}" width="${w}" height="${h}" rx="9" fill="${sh}"/>` +
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9" fill="${f}" stroke="${c}" stroke-width="1.3"/>` +
    `<text x="${x+w/2}" y="${y+h/2+4}" text-anchor="middle" font-size="12.5" font-weight="600" fill="${n.hot?'var(--accent)':'currentColor'}">${esc(n.t)}</text></g>`;
}
function flowSVG(lines, force){
  const lanes = lines.map(parseLane), H = 36, out = [];
  let S = 0; const sid = lanes.map(L => L.nodes.map(() => S++));
  let W = 0; const hl = lanes.map(L => { let x = 4; const pos = L.nodes.map((n, k) => { const w = Math.max(64, TW(n.t, 12.5) + 24); const p = {x, w}; x += w; if (k < L.labels.length) x += Math.max(40, TW(L.labels[k], 11) + 16); return p; }); W = Math.max(W, x + 4); return pos; });
  if (force || W <= AV() * 1.15){
    let y = 4;
    lanes.forEach((L, li) => {
      if (L.title){ out.push(`<text x="4" y="${y+12}" font-size="10.5" font-weight="600" fill="var(--muted)" letter-spacing=".12em" font-family="var(--mono)">${esc(L.title.toUpperCase())}</text>`); y += 20; }
      y += L.labels.some(s => s) ? 16 : 0;
      hl[li].forEach((p, k) => { out.push(box(p.x, y, p.w, H, L.nodes[k], sid[li][k]));
        if (k < L.labels.length){ const x1 = p.x + p.w + 3, x2 = hl[li][k+1].x - 3, cy = y + H/2;
          out.push(`<g class="ed" data-s="${sid[li][k+1]}"><line x1="${x1}" y1="${cy}" x2="${x2}" y2="${cy}" stroke="currentColor" stroke-width="1.3" marker-end="url(#ah)"/>` +
            (L.labels[k] ? `<text x="${(x1+x2)/2}" y="${cy-8}" text-anchor="middle" font-size="11" fill="var(--muted)">${esc(L.labels[k])}</text>` : '') + '</g>'); } });
      y += H + 18;
    });
    return {W, Hh: y - 8, body: out.join(''), steps: S};
  }
  const nw = Math.max(...lanes.flatMap(L => L.nodes.map(n => Math.max(90, TW(n.t, 12.5) + 24))));
  const lw = Math.max(0, ...lanes.flatMap(L => L.labels.map(s => TW(s, 11))));
  const cx = nw / 2 + 4; W = Math.max(nw + 8, cx + 12 + lw + 4); let y = 4;
  lanes.forEach((L, li) => {
    if (L.title){ out.push(`<text x="${cx}" y="${y+12}" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--muted)" letter-spacing=".12em" font-family="var(--mono)">${esc(L.title.toUpperCase())}</text>`); y += 20; }
    L.nodes.forEach((n, k) => { const w = Math.max(90, TW(n.t, 12.5) + 24); out.push(box(cx - w/2, y, w, H, n, sid[li][k])); y += H;
      if (k < L.labels.length){ out.push(`<g class="ed" data-s="${sid[li][k+1]}"><line x1="${cx}" y1="${y+6}" x2="${cx}" y2="${y+32}" stroke="currentColor" stroke-width="1.3" marker-end="url(#ah)"/>` +
        (L.labels[k] ? `<text x="${cx+10}" y="${y+23}" font-size="11" fill="var(--muted)">${esc(L.labels[k])}</text>` : '') + '</g>'); y += 36; } });
    y += 22;
  });
  return {W, Hh: y - 16, body: out.join(''), steps: S};
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
  msgs.forEach((m, k) => { const y = top + 18 + k * rh, x1 = X(m.a), x2 = X(m.b); let g = '';
    if (m.a === m.b){ const last = ps.indexOf(m.a) === ps.length - 1, s = last ? -1 : 1; g = `<path d="M${x1} ${y-6}h${22*s}v12h${-19*s}" fill="none" stroke="currentColor" stroke-width="1.3" marker-end="url(#ah)"/><text x="${x1+28*s}" y="${y+4}" text-anchor="${last?'end':'start'}" font-size="11" fill="var(--muted)">${esc(m.t)}</text>`; }
    else { const d = x2 > x1 ? 1 : -1;
      g = `<line x1="${x1+d*3}" y1="${y}" x2="${x2-d*3}" y2="${y}" stroke="${m.dash?'var(--muted)':'currentColor'}" stroke-width="1.3" ${m.dash?'stroke-dasharray="5 4"':''} marker-end="url(#ah)"/>` +
        `<text x="${(x1+x2)/2}" y="${y-6}" text-anchor="middle" font-size="11" fill="var(--muted)">${esc(m.t)}</text>`; }
    out.push(`<g class="ed" data-s="${k}" data-t="${esc(m.a + ' → ' + m.b + ': ' + m.t)}">${g}</g>`); });
  return {W, Hh, body: out.join(''), steps: msgs.length};
}
function layersHTML(lines){
  return '<div class="layers">' + lines.map((l, k) => { const [n, note=''] = l.split(/\s+\|\s+/); const hot = n.startsWith('*');
    return (k ? '<div class="lay-arr" aria-hidden="true">↕</div>' : '') + `<div class="lay${hot?' hot':''}" data-s="${k}" data-t="${esc(plain(n.replace(/^\*/,'')))}"><strong>${inl(n.replace(/^\*/,''))}</strong>${note?`<span>${inl(note)}</span>`:''}</div>`; }).join('') + '</div>';
}
function svgOf(d, cap, wide){ return `<svg viewBox="0 0 ${d.W} ${d.Hh}" role="img" aria-label="${esc(cap)}" style="${wide?`width:${d.W}px;max-width:none`:`width:100%;max-width:${d.W}px`};height:auto;color:var(--ink);font-family:var(--body)">${ARW}${d.body}</svg>`; }
function diagram(kind, cap, lines){
  let inner, steps = 0;
  if (kind === 'layers'){ inner = layersHTML(lines); steps = lines.length; }
  else { const d = kind === 'seq' ? seqSVG(lines) : flowSVG(lines); steps = d.steps; const wide = d.W > AV() * 1.25;
    inner = `<div class="dg-scroll">${svgOf(d, cap, wide)}</div>`; }
  if (inner.includes('max-width:none')) inner += '<div class="dg-hint">← scroll sideways to see the whole diagram →</div>';
  return `<figure class="dg" data-steps="${steps}">${inner}${cap?`<figcaption>${inl(cap)}</figcaption>`:''}</figure>`;
}
function widgetFig(name, cap){ return `<figure class="dg wg" data-w="${esc(name)}"><div class="wtag">Interactive</div><div class="wbody"></div>${cap?`<figcaption>${inl(cap)}</figcaption>`:''}</figure>`; }

// ---------- notes markup ----------
function render(src){
  const lines = src.split('\n'); let out = '', i = 0;
  while (i < lines.length){
    const L = lines[i];
    if (!L.trim()){ i++; continue; }
    if (L.startsWith('## ')){ out += `<h2>${inl(L.slice(3))}</h2>`; i++; continue; }
    if (L.startsWith('- ')){ out += '<ul>'; while (i < lines.length && lines[i].startsWith('- ')){ out += `<li>${inl(lines[i].slice(2))}</li>`; i++; } out += '</ul>'; continue; }
    if (/^%%widget\b/.test(L)){ const mm = L.match(/^%%widget\s+(\S+)\s*(.*)$/); out += widgetFig(mm[1], mm[2]); i++; continue; }
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
  mountWidgets($('#prose'));
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

// ---------- interactive widgets: helpers ----------
const WIDGETS = {};
const WX = {
  h(tag, cls, html){ const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; },
  f(n, d = 0){ if (!isFinite(n)) return '∞'; return Number(n).toLocaleString('en-US', {maximumFractionDigits: d, minimumFractionDigits: 0}); },
  bytes(b){ const u = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']; let i = 0; while (b >= 1024 && i < u.length - 1){ b /= 1024; i++; } return WX.f(b, b < 10 ? 2 : b < 100 ? 1 : 0) + ' ' + u[i]; },
  big(n){ const a = Math.abs(n); if (a >= 1e12) return WX.f(n / 1e12, 2) + 'T'; if (a >= 1e9) return WX.f(n / 1e9, 2) + 'B'; if (a >= 1e6) return WX.f(n / 1e6, 2) + 'M'; if (a >= 1e3) return WX.f(n / 1e3, 1) + 'k'; return WX.f(n, 2); },
  pct(x, d = 1){ return WX.f(x * 100, d) + '%'; },
  // controls: defs = [{k, l, min, max, step, v, fmt, opts:[[value,label]]}]
  ctl(root, defs, cb){
    const box = WX.h('div', 'wc'), vals = {};
    defs.forEach(d => {
      const lab = WX.h('label'); const id = 'w' + Math.random().toString(36).slice(2, 8);
      let inp;
      if (d.opts){ inp = WX.h('select'); d.opts.forEach(([v, t]) => { const o = WX.h('option', null, t); o.value = v; inp.appendChild(o); }); inp.value = d.v; }
      else { inp = WX.h('input'); inp.type = 'range'; inp.min = d.min; inp.max = d.max; inp.step = d.step || 1; inp.value = d.v; }
      inp.id = id;
      const top = WX.h('span'); lab.appendChild(top); lab.appendChild(inp); lab.htmlFor = id;
      const show = () => { const v = d.opts ? inp.value : +inp.value; vals[d.k] = d.opts && d.num ? +v : v; top.innerHTML = `${d.l} <b>${d.fmt ? d.fmt(vals[d.k]) : d.opts ? inp.options[inp.selectedIndex].text : WX.f(vals[d.k], 3)}</b>`; };
      inp.addEventListener('input', () => { show(); cb(vals); });
      show(); box.appendChild(lab);
    });
    root.appendChild(box); return vals;
  },
  out(el, items){ el.innerHTML = items.map(([l, v, hot]) => `<div><span>${l}</span><b class="${hot ? 'hot' : ''}">${v}</b></div>`).join(''); },
  rows(el, items, fmt = v => WX.f(v, 2)){
    const mx = Math.max(1e-12, ...items.map(i => i.max != null ? i.max : i.v));
    el.innerHTML = items.map(i => `<div class="wrow${i.off ? ' off' : ''}${i.hot ? ' hot' : ''}"><span class="l" title="${esc(i.l)}">${esc(i.l)}</span><span><i class="wbar" style="display:block;width:${Math.max(0, Math.min(100, i.v / (i.max != null ? i.max : mx) * 100))}%;${i.c ? 'background:' + i.c : ''}"></i></span><span class="v">${i.t != null ? i.t : fmt(i.v)}</span></div>`).join('');
  },
  // line chart: series [{pts:[[x,y]], c, dash, l}], opts {xl, yl, xlog, ylog, ymin, ymax, xmin, xmax, marks:[{x, l}], fx, fy}
  line(el, series, o = {}){
    const W = 560, H = 230, L = 52, R = 14, T = 14, B = 38;
    const all = series.flatMap(s => s.pts);
    const tx = v => o.xlog ? Math.log10(v) : v, ty = v => o.ylog ? Math.log10(Math.max(v, 1e-12)) : v;
    let x0 = o.xmin != null ? o.xmin : Math.min(...all.map(p => p[0])), x1 = o.xmax != null ? o.xmax : Math.max(...all.map(p => p[0]));
    let y0 = o.ymin != null ? o.ymin : Math.min(0, ...all.map(p => p[1])), y1 = o.ymax != null ? o.ymax : Math.max(...all.map(p => p[1])) * 1.08 || 1;
    if (o.ylog && !(y0 > 0)) y0 = Math.min(...all.map(p => p[1]).filter(v => v > 0)); if (o.xlog && !(x0 > 0)) x0 = Math.min(...all.map(p => p[0]).filter(v => v > 0));
    const X = v => L + (tx(v) - tx(x0)) / ((tx(x1) - tx(x0)) || 1) * (W - L - R), Y = v => T + (1 - (ty(v) - ty(y0)) / ((ty(y1) - ty(y0)) || 1)) * (H - T - B);
    const fx = o.fx || (v => WX.big(v)), fy = o.fy || (v => WX.big(v));
    let g = '';
    const ticks = (a, b, log) => { if (log){ const r = []; const e0 = Math.ceil(Math.log10(a)), e1 = Math.floor(Math.log10(b)), st = Math.max(1, Math.ceil((e1 - e0 + 1) / 7)); if (!isFinite(e0) || !isFinite(e1)) return r; for (let e = e0; e <= e1; e += st) r.push(10 ** e); return r; } const n = 5, s = (b - a) / n; return Array.from({length: n + 1}, (_, i) => a + i * s); };
    ticks(y0, y1, o.ylog).forEach(v => { const y = Y(v); g += `<line class="gr" x1="${L}" x2="${W - R}" y1="${y}" y2="${y}"/><text x="${L - 6}" y="${y + 4}" text-anchor="end">${fy(v)}</text>`; });
    ticks(x0, x1, o.xlog).forEach(v => { const x = X(v); g += `<text x="${x}" y="${H - B + 16}" text-anchor="middle">${fx(v)}</text>`; });
    g += `<line class="ax" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/><line class="ax" x1="${L}" x2="${L}" y1="${T}" y2="${H - B}"/>`;
    if (o.xl) g += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">${o.xl}</text>`;
    if (o.yl) g += `<text x="12" y="${(T + H - B) / 2}" text-anchor="middle" transform="rotate(-90 12 ${(T + H - B) / 2})">${o.yl}</text>`;
    (o.marks || []).forEach(m => { const x = X(m.x); g += `<line x1="${x}" x2="${x}" y1="${T}" y2="${H - B}" stroke="var(--accent)" stroke-dasharray="3 3"/>${m.l ? `<text x="${x + 4}" y="${T + 10}" style="fill:var(--accent)">${m.l}</text>` : ''}`; });
    series.forEach(s => { if (!s.pts.length) return; g += `<polyline fill="none" stroke="${s.c || 'var(--accent)'}" stroke-width="${s.w || 2.2}" ${s.dash ? 'stroke-dasharray="5 4"' : ''} points="${s.pts.map(p => X(p[0]).toFixed(1) + ',' + Y(p[1]).toFixed(1)).join(' ')}"/>`;
      (s.dots || []).forEach(p => { g += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="4.5" fill="${s.c || 'var(--accent)'}"/>`; }); });
    const leg = series.filter(s => s.l).map(s => `<span style="display:inline-flex;align-items:center;gap:6px;margin-right:14px"><i style="width:14px;height:3px;border-radius:2px;background:${s.c || 'var(--accent)'};display:inline-block"></i>${s.l}</span>`).join('');
    el.innerHTML = `<div class="wscroll"><svg class="wsvg" viewBox="0 0 ${W} ${H}" style="min-width:420px">${g}</svg></div>${leg ? `<div class="wnote">${leg}</div>` : ''}`;
  },
  softmax(z, T = 1){ const m = Math.max(...z.map(v => v / T)); const e = z.map(v => Math.exp(v / T - m)); const s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); },
  rng(seed){ let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; },
  gauss(r){ let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); },
  C1: 'var(--accent)', C2: 'var(--flame)', C3: 'var(--go)', C4: 'var(--gold)', CM: 'var(--muted)',
};
function mountWidgets(scope){
  (scope || document).querySelectorAll('.wg[data-w]').forEach(fig => {
    if (fig.dataset.on) return; fig.dataset.on = '1';
    const fn = WIDGETS[fig.dataset.w], body = fig.querySelector('.wbody');
    if (!fn){ body.innerHTML = '<div class="wnote">This figure failed to load.</div>'; return; }
    try { fn(body); } catch(e){ body.innerHTML = '<div class="wnote">This figure failed to load.</div>'; console.error(e); }
  });
  (scope || document).querySelectorAll('.dg[data-steps]').forEach(stepper);
}
// step-through for flow / seq / layers diagrams
function stepper(fig){
  if (fig.dataset.on) return; fig.dataset.on = '1';
  const n = +fig.dataset.steps; if (n < 2) return;
  let i = -1, timer = null;
  const ctl = WX.h('div', 'dgc', `<button type="button" class="wb" data-a="play">▶ Play</button><button type="button" class="wb" data-a="prev">‹ Back</button><button type="button" class="wb" data-a="next">Step ›</button><button type="button" class="wb" data-a="reset">Show all</button><span class="st">Step through the figure</span>`);
  const cap = fig.querySelector('figcaption'); fig.insertBefore(ctl, cap);
  const st = ctl.querySelector('.st');
  const show = () => {
    fig.classList.toggle('stepping', i >= 0);
    fig.querySelectorAll('.nd[data-s], .ed[data-s]').forEach(e => e.classList.toggle('on', i >= 0 && +e.dataset.s <= i));
    fig.querySelectorAll('.lay[data-s]').forEach(e => e.classList.toggle('on', i >= 0 && +e.dataset.s === i));
    st.textContent = i < 0 ? 'Step through the figure' : `Step ${i + 1} of ${n}`;
    const lbl = i >= 0 ? fig.querySelector(`[data-s="${i}"][data-t]`) : null; if (lbl) st.textContent += ' · ' + lbl.dataset.t;
  };
  const stop = () => { clearInterval(timer); timer = null; ctl.querySelector('[data-a=play]').textContent = '▶ Play'; };
  ctl.addEventListener('click', e => { const a = e.target.closest('[data-a]')?.dataset.a; if (!a) return;
    if (a === 'next'){ stop(); i = Math.min(n - 1, i + 1); }
    if (a === 'prev'){ stop(); i = Math.max(-1, i - 1); }
    if (a === 'reset'){ stop(); i = -1; }
    if (a === 'play'){ if (timer){ stop(); } else { if (i >= n - 1) i = -1; e.target.textContent = '❚❚ Pause'; timer = setInterval(() => { i++; show(); if (i >= n - 1) stop(); }, RM ? 1400 : 900); } }
    show(); });
  fig.addEventListener('click', e => { const s = e.target.closest('.nd[data-s], .lay[data-s]'); if (!s) return; stop(); i = +s.dataset.s; show(); });
}
// ---------- module 1 ----------
WIDGETS.tokenizer = root => {
  root.insertAdjacentHTML('beforeend', `<textarea id="tk-in" aria-label="Text to tokenize">Tokenization isn't free: "unbelievably" costs more than "cat", and 2026-11-12 or résumé split into pieces.</textarea><div class="wtok"></div><div class="wout"></div><div class="wnote">An approximate BPE-style splitter, built for intuition. Real tokenizers use a learned vocabulary of 50k to 200k pieces, so exact counts differ.</div>`);
  const ta = root.querySelector('textarea'), tk = root.querySelector('.wtok'), out = root.querySelector('.wout');
  const SUF = ['ization', 'ation', 'ness', 'ment', 'able', 'ably', 'ing', 'ion', 'ers', 'est', 'ed', 'ly', 'er', 's'];
  const split = w => { const p = []; let s = w; const suf = SUF.find(x => s.length > x.length + 2 && s.toLowerCase().endsWith(x)); let tail = ''; if (suf){ tail = s.slice(-suf.length); s = s.slice(0, -suf.length); }
    while (s.length > 5){ p.push(s.slice(0, 4)); s = s.slice(4); } if (s) p.push(s); if (tail) p.push(tail); return p; };
  const run = () => {
    const toks = []; const re = /(\s?)([A-Za-z]+|[0-9]+|[^\sA-Za-z0-9])/g; let m;
    while ((m = re.exec(ta.value))){ const sp = m[1], w = m[2];
      if (/^[0-9]+$/.test(w)){ for (let i = 0; i < w.length; i += 3) toks.push((i ? '' : sp) + w.slice(i, i + 3)); }
      else if (/^[A-Za-z]+$/.test(w)){ const ps = w.length <= 6 ? [w] : split(w); ps.forEach((x, i) => toks.push((i ? '' : sp) + x)); }
      else { const n = w.charCodeAt(0) > 127 ? 2 : 1; for (let i = 0; i < n; i++) toks.push((i ? '' : sp) + (n > 1 ? (i ? '·' : w) : w)); } }
    const cols = ['var(--accent-soft)', 'var(--go-soft)', 'var(--gold-soft)', 'var(--flame-soft)'];
    tk.innerHTML = toks.map((t, i) => `<span style="background:${cols[i % 4]}">${esc(t)}</span>`).join('');
    const chars = ta.value.length, words = (ta.value.match(/\S+/g) || []).length;
    WX.out(out, [['Characters', WX.f(chars)], ['Words', WX.f(words)], ['Tokens', WX.f(toks.length), 1], ['Chars per token', WX.f(chars / Math.max(1, toks.length), 2)], ['Tokens per word', WX.f(toks.length / Math.max(1, words), 2)]]);
  };
  ta.addEventListener('input', run); run();
};
WIDGETS.latency = root => {
  const bar = WX.h('div'), out = WX.h('div', 'wout'), note = WX.h('div', 'wnote');
  const go = v => {
    const ttft = v.p / v.pf + v.q / 1000, dec = v.o / v.ds, tot = ttft + dec;
    bar.innerHTML = `<div style="display:flex;height:26px;border-radius:7px;overflow:hidden;font:500 11px var(--mono)"><div style="width:${ttft / tot * 100}%;background:var(--gold);color:#fff;display:flex;align-items:center;padding-left:6px;white-space:nowrap;overflow:hidden">prefill</div><div style="flex:1;background:var(--accent);color:var(--accent-ink);display:flex;align-items:center;padding-left:6px;white-space:nowrap;overflow:hidden">decode ${WX.f(v.o)} tokens</div></div>`;
    WX.out(out, [['Time to first token', WX.f(ttft, 2) + ' s', 1], ['Decode time', WX.f(dec, 2) + ' s'], ['Total', WX.f(tot, 2) + ' s', 1], ['Share spent decoding', WX.pct(dec / tot, 0)]]);
    note.textContent = dec > ttft * 3 ? 'Output length dominates. Shorter answers or faster decode (quantization, speculative decoding) help most.' : ttft > dec ? 'The prompt dominates. Prompt caching or sending less context helps most.' : 'Prefill and decode both matter here.';
  };
  const v = WX.ctl(root, [
    {k: 'p', l: 'Prompt tokens', min: 100, max: 100000, step: 100, v: 2000},
    {k: 'o', l: 'Output tokens', min: 10, max: 4000, step: 10, v: 300},
    {k: 'pf', l: 'Prefill speed (tok/s)', min: 1000, max: 30000, step: 500, v: 10000},
    {k: 'ds', l: 'Decode speed (tok/s per user)', min: 5, max: 200, step: 1, v: 40},
    {k: 'q', l: 'Queueing + network (ms)', min: 0, max: 2000, step: 10, v: 150}], go);
  root.append(bar, out, note); go(v);
};
WIDGETS.stack = root => {
  const out = WX.h('div', 'wout'), rows = WX.h('div'), note = WX.h('div', 'wnote', 'Illustrative prices: large model $3 / $15 per million input / output tokens, small model $0.25 / $1.25. Change the knobs to see which layer pays for itself.');
  const go = v => {
    const inTok = 800 + v.rag * 500, outTok = 350;
    const big = (inTok * 3 + outTok * 15) / 1e6, small = (inTok * 0.25 + outTok * 1.25) / 1e6;
    const routed = v.router / 100, hit = v.cache / 100;
    const modelCost = (1 - routed) * big + routed * small, cost = (1 - hit) * modelCost + (v.guard ? (1 - hit) * small * 0.3 : 0);
    const lat = (1 - hit) * ((1 - routed) * 6.0 + routed * 1.8 + v.rag * 0.08 + (v.guard ? 0.35 : 0)) + hit * 0.05;
    WX.out(out, [['Cost per 1k requests', '$' + WX.f(cost * 1000, 2), 1], ['Avg latency', WX.f(lat, 2) + ' s', 1], ['Input tokens / call', WX.f(inTok)], ['Baseline (big model only)', '$' + WX.f(big * 1000, 2)]]);
    WX.rows(rows, [{l: 'Cache hits', v: hit, max: 1, t: WX.pct(hit, 0)}, {l: 'To small model', v: routed * (1 - hit), max: 1, t: WX.pct(routed * (1 - hit), 0)}, {l: 'To large model', v: (1 - routed) * (1 - hit), max: 1, t: WX.pct((1 - routed) * (1 - hit), 0)}]);
  };
  const v = WX.ctl(root, [
    {k: 'cache', l: 'Cache hit rate', min: 0, max: 80, v: 0, fmt: x => x + '%'},
    {k: 'router', l: 'Routed to small model', min: 0, max: 90, v: 0, fmt: x => x + '%'},
    {k: 'rag', l: 'Retrieved chunks per call', min: 0, max: 20, v: 0},
    {k: 'guard', l: 'Guardrail model', opts: [['0', 'Off'], ['1', 'On']], v: '0', num: 1}], go);
  root.append(rows, out, note); go(v);
};
// ---------- module 2 ----------
WIDGETS.analogy = root => {
  const P = {king: [3, 4], queen: [1, 5.2], man: [3.2, 1], woman: [1.2, 2.2], Paris: [-3, 1.5], France: [-4.6, 3.8], Rome: [-1.6, 0.6], Italy: [-3.2, 2.9], walk: [4.4, -3], walked: [2.8, -4.4], swim: [5.6, -1.6], swam: [4.0, -3.0]};
  const A = [['king', 'man', 'woman', 'queen'], ['France', 'Paris', 'Rome', 'Italy'], ['walked', 'walk', 'swim', 'swam']];
  const btns = WX.h('div', 'wbtns', A.map((a, i) => `<button type="button" class="wb${i ? '' : ' on'}" data-i="${i}">${a[0]} − ${a[1]} + ${a[2]}</button>`).join(''));
  const fig = WX.h('div'), out = WX.h('div', 'wnote');
  const S = 34, ox = 300, oy = 190, X = x => ox + x * S, Y = y => oy - y * S;
  const draw = i => {
    const [a, b, c, want] = A[i], r = [P[a][0] - P[b][0] + P[c][0], P[a][1] - P[b][1] + P[c][1]];
    let best = null, bd = 1e9; Object.entries(P).forEach(([k, p]) => { if ([a, b, c].includes(k)) return; const d = Math.hypot(p[0] - r[0], p[1] - r[1]); if (d < bd){ bd = d; best = k; } });
    let g = `<line class="gr" x1="0" x2="600" y1="${oy}" y2="${oy}"/><line class="gr" x1="${ox}" x2="${ox}" y1="0" y2="380"/>`;
    const arrow = (p, q, c, lab) => `<line x1="${X(p[0])}" y1="${Y(p[1])}" x2="${X(q[0])}" y2="${Y(q[1])}" stroke="${c}" stroke-width="2" marker-end="url(#wa)"/>${lab ? `<text x="${(X(p[0]) + X(q[0])) / 2 + 6}" y="${(Y(p[1]) + Y(q[1])) / 2 - 6}" style="fill:${c}">${lab}</text>` : ''}`;
    g += arrow(P[b], P[a], 'var(--muted)', `${a} − ${b}`) + arrow(P[c], r, 'var(--accent)', 'same offset');
    Object.entries(P).forEach(([k, p]) => { const on = [a, b, c].includes(k), hit = k === best; g += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="${hit ? 7 : 5}" fill="${hit ? 'var(--go)' : on ? 'var(--ink)' : 'var(--line2)'}"/><text x="${X(p[0]) + 9}" y="${Y(p[1]) + 4}" style="fill:${hit ? 'var(--go)' : on ? 'var(--ink)' : 'var(--muted)'};font-weight:${hit || on ? 600 : 400}">${k}</text>`; });
    g += `<circle cx="${X(r[0])}" cy="${Y(r[1])}" r="9" fill="none" stroke="var(--accent)" stroke-width="2" stroke-dasharray="3 3"/>`;
    fig.innerHTML = `<div class="wscroll"><svg class="wsvg" viewBox="0 0 600 380" style="min-width:420px"><defs><marker id="wa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="context-stroke"/></marker></defs>${g}</svg></div>`;
    out.innerHTML = `Result vector lands nearest <b style="color:var(--go)">${best}</b> (distance ${WX.f(bd, 2)}). Expected: ${want}. Toy 2-D coordinates chosen by hand; real embeddings do this in hundreds of dimensions, and analogies work less cleanly than this picture suggests.`;
  };
  btns.addEventListener('click', e => { const b = e.target.closest('[data-i]'); if (!b) return; btns.querySelectorAll('.wb').forEach(x => x.classList.toggle('on', x === b)); draw(+b.dataset.i); });
  root.append(btns, fig, out); draw(0);
};
WIDGETS.simspace = root => {
  const D = [['Reset your password', 2.6, 0.8], ['Recover account access', 1.4, 0.45], ['Change login email', 2.2, 1.6], ['Two-factor setup', 1.0, 1.1], ['Refund policy', -1.8, 1.9], ['Cancel subscription', -2.4, 0.9], ['Invoice download', -1.1, 2.6], ['Shipping times', -2.2, -1.6], ['Track my order', -1.2, -2.6], ['Return a damaged item', -0.5, -1.2], ['API rate limits', 2.0, -2.2], ['Webhook retries', 0.9, -1.9]];
  const fig = WX.h('div'), list = WX.h('div'), hint = WX.h('div', 'wnote', 'Drag the purple query point. Documents have different vector lengths, so dot product and cosine can disagree unless you normalize.');
  let q = [2.1, 0.6];
  const S = 52, ox = 300, oy = 200;
  const v = WX.ctl(root, [{k: 'm', l: 'Metric', opts: [['cos', 'Cosine similarity'], ['dot', 'Dot product'], ['l2', 'Euclidean (L2) distance']], v: 'cos'}, {k: 'n', l: 'Normalize vectors', opts: [['0', 'No'], ['1', 'Yes']], v: '0', num: 1}], () => draw());
  const draw = () => {
    const nz = p => { const l = Math.hypot(p[0], p[1]) || 1; return v.n ? [p[0] / l, p[1] / l] : p; };
    const qq = nz(q);
    const sc = D.map(([t, x, y]) => { const d = nz([x, y]); const dot = d[0] * qq[0] + d[1] * qq[1]; const cos = dot / (Math.hypot(...d) * Math.hypot(...qq) || 1); const l2 = Math.hypot(d[0] - qq[0], d[1] - qq[1]); return {t, d, s: v.m === 'cos' ? cos : v.m === 'dot' ? dot : -l2, raw: v.m === 'cos' ? cos : v.m === 'dot' ? dot : l2}; });
    const rank = [...sc].sort((a, b) => b.s - a.s), top = new Set(rank.slice(0, 3).map(r => r.t));
    let g = `<line class="gr" x1="0" x2="600" y1="${oy}" y2="${oy}"/><line class="gr" x1="${ox}" x2="${ox}" y1="0" y2="400"/>`;
    if (v.n) g += `<circle cx="${ox}" cy="${oy}" r="${S}" fill="none" stroke="var(--line2)" stroke-dasharray="4 4"/>`;
    sc.forEach(r => { const X = ox + r.d[0] * S, Y = oy - r.d[1] * S, on = top.has(r.t); g += `<line x1="${ox}" y1="${oy}" x2="${X}" y2="${Y}" stroke="${on ? 'var(--go)' : 'var(--line)'}" stroke-width="1.2"/><circle cx="${X}" cy="${Y}" r="5" fill="${on ? 'var(--go)' : 'var(--line2)'}"/><text x="${X + 8}" y="${Y + 4}" style="fill:${on ? 'var(--go)' : 'var(--muted)'};font-weight:${on ? 600 : 400}">${r.t}</text>`; });
    const QX = ox + qq[0] * S, QY = oy - qq[1] * S;
    g += `<line x1="${ox}" y1="${oy}" x2="${QX}" y2="${QY}" stroke="var(--accent)" stroke-width="2"/><circle class="qh" cx="${QX}" cy="${QY}" r="10" fill="var(--accent)" stroke="var(--card)" stroke-width="3" style="cursor:grab"/><text x="${QX + 12}" y="${QY - 10}" style="fill:var(--accent);font-weight:600">query</text>`;
    svg.innerHTML = g;
    WX.rows(list, rank.slice(0, 5).map((r, i) => ({l: `${i + 1}. ${r.t}`, v: v.m === 'l2' ? 1 / (1 + r.raw) : Math.max(0, r.raw), max: v.m === 'cos' ? 1 : undefined, t: WX.f(r.raw, 3), hot: i < 3})));
  };
  fig.innerHTML = '<div class="wscroll"><svg class="wsvg" viewBox="0 0 600 400" style="min-width:420px;touch-action:none;cursor:grab"></svg></div>';
  const svg = fig.querySelector('svg');
  const at = e => { const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY; const p = pt.matrixTransform(svg.getScreenCTM().inverse()); q = [(p.x - ox) / S, (oy - p.y) / S]; draw(); };
  svg.addEventListener('pointerdown', e => { svg.setPointerCapture(e.pointerId); at(e); });
  svg.addEventListener('pointermove', e => { if (svg.hasPointerCapture(e.pointerId)) at(e); });
  root.append(fig, list, hint); draw();
};
WIDGETS.vecmem = root => {
  const out = WX.h('div', 'wout'), note = WX.h('div', 'wnote');
  const go = v => {
    const N = 10 ** v.n, bpv = v.t === 'pq' ? 64 : v.d * {f32: 4, f16: 2, i8: 1, b1: 1 / 8}[v.t];
    const over = v.ix === 'hnsw' ? 16 * 2 * 4 : v.ix === 'ivf' ? 8 : 0;
    const tot = N * (bpv + over), mach = Math.ceil(tot / (v.ram * 1024 ** 3 * 0.7));
    WX.out(out, [['Vectors', WX.big(N)], ['Bytes per vector', WX.f(bpv + over, 1)], ['Index in memory', WX.bytes(tot), 1], ['Machines at 70% RAM', WX.f(mach), 1]]);
    note.textContent = tot < 50 * 1024 ** 3 ? 'Fits on one ordinary server. A Postgres extension or a single-node index is enough.' : mach <= 4 ? 'A large single node or a few shards. Compression starts to pay off.' : 'Needs sharding across many machines, or compression and disk-based indexes. This is where dedicated vector databases earn their keep.';
  };
  const v = WX.ctl(root, [
    {k: 'n', l: 'Number of vectors', min: 4, max: 10, step: 0.1, v: 6.3, fmt: x => WX.big(10 ** x)},
    {k: 'd', l: 'Dimensions', opts: [['384', '384'], ['768', '768'], ['1024', '1024'], ['1536', '1536'], ['3072', '3072']], v: '768', num: 1},
    {k: 't', l: 'Storage format', opts: [['f32', 'float32'], ['f16', 'float16'], ['i8', 'int8'], ['b1', 'binary (1 bit)'], ['pq', 'PQ, 64 bytes']], v: 'f32'},
    {k: 'ix', l: 'Index overhead', opts: [['none', 'None (flat)'], ['hnsw', 'HNSW, M = 16'], ['ivf', 'IVF lists']], v: 'hnsw'},
    {k: 'ram', l: 'RAM per machine (GB)', min: 16, max: 1024, step: 16, v: 128}], go);
  root.append(out, note); go(v);
};
WIDGETS.pq = root => {
  const strip = WX.h('div'), out = WX.h('div', 'wout');
  const go = v => {
    const m = v.m, sub = v.d / m, codeB = m * v.b / 8, raw = v.d * 4, cb = m * 2 ** v.b * sub * 4;
    const cols = ['var(--accent)', 'var(--go)', 'var(--gold)', 'var(--flame)'];
    const shown = Math.min(m, 48);
    strip.innerHTML = `<div class="wnote" style="margin-bottom:6px">${v.d} floats split into ${m} sub-vectors of ${WX.f(sub, 2)} dims. Each becomes one ${v.b}-bit code (an id among ${2 ** v.b} centroids).</div><div style="display:flex;gap:2px">${Array.from({length: shown}, (_, i) => `<i style="flex:1;height:22px;border-radius:3px;background:${cols[i % 4]};opacity:.85"></i>`).join('')}</div>${m > shown ? `<div class="wnote">showing ${shown} of ${m} segments</div>` : ''}`;
    WX.out(out, [['Raw float32', WX.bytes(raw)], ['PQ code', WX.f(codeB, 1) + ' B', 1], ['Compression', WX.f(raw / codeB, 1) + '×', 1], ['Codebooks (shared)', WX.bytes(cb)], ['1B vectors, raw', WX.bytes(raw * 1e9)], ['1B vectors, PQ', WX.bytes(codeB * 1e9), 1]]);
    if (!Number.isInteger(sub)) out.insertAdjacentHTML('beforeend', '<div><span>Warning</span><b>m must divide D</b></div>');
  };
  const v = WX.ctl(root, [{k: 'd', l: 'Dimensions D', opts: [['384', '384'], ['768', '768'], ['1024', '1024'], ['1536', '1536']], v: '768', num: 1}, {k: 'm', l: 'Sub-vectors m', opts: [['8', '8'], ['16', '16'], ['32', '32'], ['48', '48'], ['64', '64'], ['96', '96'], ['128', '128']], v: '96', num: 1}, {k: 'b', l: 'Bits per code', opts: [['4', '4 (16 centroids)'], ['8', '8 (256 centroids)']], v: '8', num: 1}], go);
  root.append(strip, out); go(v);
};
WIDGETS.binq = root => {
  // real computation: 3000 clustered random vectors, 64 dims, measure recall@10 of quantized search with rescoring
  const R = WX.rng(7), d = 64, N = 3000, C = 30;
  const cents = Array.from({length: C}, () => Array.from({length: d}, () => WX.gauss(R)));
  const norm = a => { const l = Math.hypot(...a) || 1; return a.map(x => x / l); };
  const V = Array.from({length: N}, () => { const c = cents[Math.floor(R() * C)]; return norm(c.map(x => x + WX.gauss(R) * 0.9)); });
  const Q = Array.from({length: 40}, () => { const c = cents[Math.floor(R() * C)]; return norm(c.map(x => x + WX.gauss(R) * 0.9)); });
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  let lo = Infinity, hi = -Infinity; V.forEach(v => v.forEach(x => { lo = Math.min(lo, x); hi = Math.max(hi, x); }));
  const i8 = V.map(v => v.map(x => Math.round((x - lo) / (hi - lo) * 255)));
  const deq = q => q.map(x => x / 255 * (hi - lo) + lo);
  const b1 = V.map(v => v.map(x => x > 0 ? 1 : -1)), i8d = i8.map(deq);
  const truth = Q.map(q => V.map((v, i) => [dot(q, v), i]).sort((a, b) => b[0] - a[0]).slice(0, 10).map(x => x[1]));
  const out = WX.h('div', 'wout'), bars = WX.h('div'), note = WX.h('div', 'wnote', 'Computed live in your browser: 3,000 random clustered 64-dim vectors, 40 queries, recall@10 against exact float32 search.');
  const go = v => {
    const k = 10 * v.os; let hit = 0;
    Q.forEach((q, qi) => {
      const qb = q.map(x => x > 0 ? 1 : -1);
      const s = v.t === 'i8' ? i8d.map((c, i) => [dot(q, c), i]) : b1.map((c, i) => [dot(qb, c), i]);
      let cand = s.sort((a, b) => b[0] - a[0]).slice(0, k).map(x => x[1]);
      if (v.rs) cand = cand.map(i => [dot(q, V[i]), i]).sort((a, b) => b[0] - a[0]).map(x => x[1]);
      const top = new Set(cand.slice(0, 10)); hit += truth[qi].filter(i => top.has(i)).length;
    });
    const rec = hit / (Q.length * 10), bpv = v.t === 'i8' ? d : d / 8;
    WX.out(out, [['Recall@10', WX.pct(rec), 1], ['Bytes per vector', WX.f(bpv) + ' (float32: ' + d * 4 + ')'], ['Memory saved', WX.f(d * 4 / bpv) + '×'], ['Candidates rescored', v.rs ? WX.f(k) : '0']]);
    WX.rows(bars, [{l: 'float32', v: 1, max: 1, t: '100%'}, {l: v.t === 'i8' ? 'int8' : 'binary', v: rec, max: 1, t: WX.pct(rec), hot: 1}]);
  };
  const v = WX.ctl(root, [{k: 't', l: 'Quantization', opts: [['b1', 'Binary (1 bit)'], ['i8', 'int8 scalar']], v: 'b1'}, {k: 'os', l: 'Oversampling factor', min: 1, max: 10, v: 1, fmt: x => x + '× (' + x * 10 + ' candidates)'}, {k: 'rs', l: 'Rescore with float32', opts: [['0', 'No'], ['1', 'Yes']], v: '0', num: 1}], go);
  root.append(bars, out, note); go(v);
};
WIDGETS.flatcost = root => {
  const out = WX.h('div', 'wout');
  const go = v => { const N = 10 ** v.n, bytes = N * v.d * 4, t = bytes / (v.bw * 1e9), flops = N * v.d * 2;
    WX.out(out, [['Data scanned per query', WX.bytes(bytes)], ['Multiply-adds', WX.big(flops / 2)], ['Time per query (bandwidth-bound)', t < 1 ? WX.f(t * 1000, 1) + ' ms' : WX.f(t, 2) + ' s', 1], ['Max QPS per machine', WX.f(1 / t, 1), 1]]); };
  const v = WX.ctl(root, [{k: 'n', l: 'Vectors', min: 4, max: 9.5, step: 0.1, v: 7.7, fmt: x => WX.big(10 ** x)}, {k: 'd', l: 'Dimensions', min: 128, max: 3072, step: 128, v: 768}, {k: 'bw', l: 'Memory bandwidth (GB/s)', min: 20, max: 3000, step: 10, v: 200}], go);
  root.append(out); go(v);
};
WIDGETS.rrf = root => {
  const A = ['Doc E (exact error code)', 'Doc B', 'Doc G', 'Doc A', 'Doc H', 'Doc C'], B = ['Doc A', 'Doc B', 'Doc C', 'Doc D', 'Doc F', 'Doc E (exact error code)'];
  const tbl = WX.h('div');
  const go = v => {
    const s = {}; [A, B].forEach(L => L.forEach((d, r) => { s[d] = (s[d] || 0) + 1 / (v.k + r + 1); }));
    const rk = Object.entries(s).sort((a, b) => b[1] - a[1]);
    tbl.innerHTML = `<div class="wscroll"><table class="wtbl"><tr><th>#</th><th>Keyword (BM25)</th><th>Vector</th><th>Fused (RRF)</th><th>Score</th></tr>${rk.map(([d, sc], i) => `<tr class="${i < 3 ? 'hot' : ''}"><td class="n">${i + 1}</td><td>${A[i] || ''}</td><td>${B[i] || ''}</td><td><b>${d}</b></td><td class="n">${sc.toFixed(4)}</td></tr>`).join('')}</table></div><div class="wnote">score(doc) = Σ 1 / (k + rank). Small k rewards being at the very top of either list; large k rewards appearing in both.</div>`;
  };
  const v = WX.ctl(root, [{k: 'k', l: 'RRF constant k', min: 0, max: 100, v: 60}], go);
  root.append(tbl); go(v);
};
// k-means 2D points used by ivf and hnsw
function pts2d(seed, n){ const R = WX.rng(seed), cs = Array.from({length: 9}, () => [R() * 520 + 40, R() * 300 + 30]); return Array.from({length: n}, () => { const c = cs[Math.floor(R() * cs.length)]; return [c[0] + WX.gauss(R) * 38, c[1] + WX.gauss(R) * 30]; }).filter(p => p[0] > 4 && p[0] < 596 && p[1] > 4 && p[1] < 356); }
WIDGETS.ivf = root => {
  const P = pts2d(11, 700), K = 16, R = WX.rng(3);
  let C = Array.from({length: K}, () => P[Math.floor(R() * P.length)].slice()), as = [];
  for (let it = 0; it < 12; it++){ as = P.map(p => { let b = 0, bd = 1e9; C.forEach((c, j) => { const d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2; if (d < bd){ bd = d; b = j; } }); return b; }); C = C.map((c, j) => { const m = P.filter((_, i) => as[i] === j); return m.length ? [m.reduce((a, p) => a + p[0], 0) / m.length, m.reduce((a, p) => a + p[1], 0) / m.length] : c; }); }
  let q = [300, 180];
  const fig = WX.h('div'), out = WX.h('div', 'wout'), hint = WX.h('div', 'wnote', 'Click anywhere to move the query. Squares are cluster centroids; highlighted points are the lists actually scanned.');
  const cols = ['#5A4FCF', '#1F7A72', '#C99A3A', '#C2577A', '#6D8FD6', '#7A8B3A', '#B5651D', '#3A8FA0'];
  const go = () => {
    const dc = C.map((c, j) => [(c[0] - q[0]) ** 2 + (c[1] - q[1]) ** 2, j]).sort((a, b) => a[0] - b[0]).slice(0, v.np).map(x => x[1]), probe = new Set(dc);
    const dist = P.map((p, i) => [(p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2, i]);
    const truth = [...dist].sort((a, b) => a[0] - b[0]).slice(0, 10).map(x => x[1]);
    const found = dist.filter(([, i]) => probe.has(as[i])).sort((a, b) => a[0] - b[0]).slice(0, 10).map(x => x[1]);
    const fs = new Set(found), ts = new Set(truth), scanned = P.filter((_, i) => probe.has(as[i])).length;
    let g = '';
    P.forEach((p, i) => { const on = probe.has(as[i]); g += `<circle cx="${p[0]}" cy="${p[1]}" r="${ts.has(i) ? 4.5 : 2.6}" fill="${ts.has(i) ? (fs.has(i) ? 'var(--go)' : 'var(--bad)') : on ? cols[as[i] % cols.length] : 'var(--line2)'}" opacity="${on || ts.has(i) ? 1 : .55}"/>`; });
    C.forEach((c, j) => { g += `<rect x="${c[0] - 5}" y="${c[1] - 5}" width="10" height="10" fill="${probe.has(j) ? 'var(--ink)' : 'none'}" stroke="var(--ink)" stroke-width="1.5"/>`; });
    g += `<circle cx="${q[0]}" cy="${q[1]}" r="8" fill="var(--accent)" stroke="var(--card)" stroke-width="3"/>`;
    fig.innerHTML = `<div class="wscroll"><svg class="wsvg" viewBox="0 0 600 360" style="min-width:420px;cursor:crosshair">${g}</svg></div><div class="wnote"><span style="color:var(--go)">●</span> true neighbour found &nbsp; <span style="color:var(--bad)">●</span> true neighbour missed</div>`;
    const svg = fig.querySelector('svg'); svg.onclick = e => { const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY; const p = pt.matrixTransform(svg.getScreenCTM().inverse()); q = [p.x, p.y]; go(); };
    WX.out(out, [['Recall@10', WX.pct(fs.size ? truth.filter(i => fs.has(i)).length / 10 : 0, 0), 1], ['Points scanned', `${scanned} of ${P.length}`], ['Share of data scanned', WX.pct(scanned / P.length, 0), 1]]);
  };
  const v = WX.ctl(root, [{k: 'np', l: 'nprobe (clusters scanned)', min: 1, max: K, v: 2, fmt: x => `${x} of ${K}`}], go);
  root.append(fig, out, hint); go();
};
WIDGETS.hnsw = root => {
  const P = pts2d(5, 260), R = WX.rng(9), M = 5;
  const lvl = P.map(() => { let l = 0; while (R() < 0.25 && l < 2) l++; return l; });
  const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
  const nbr = [0, 1, 2].map(L => P.map((p, i) => lvl[i] < L ? [] : P.map((q, j) => [d2(p, q), j]).filter(([, j]) => j !== i && lvl[j] >= L).sort((a, b) => a[0] - b[0]).slice(0, L ? 3 : M).map(x => x[1])));
  const entry = lvl.indexOf(2) >= 0 ? lvl.indexOf(2) : lvl.indexOf(1);
  let q = [470, 290], path = [], step = 0;
  const search = () => { path = []; let cur = entry; for (let L = 2; L >= 0; L--){ if (lvl[cur] < L) continue; let moved = true; while (moved){ moved = false; path.push({L, n: cur}); for (const j of nbr[L][cur]) if (d2(P[j], q) < d2(P[cur], q)){ cur = j; moved = true; } } } let best = 0; P.forEach((p, i) => { if (d2(p, q) < d2(P[best], q)) best = i; }); return best; };
  let truth = search();
  const fig = WX.h('div'), out = WX.h('div', 'wnote'), btns = WX.h('div', 'wbtns', '<button type="button" class="wb" data-a="s">Step ›</button><button type="button" class="wb" data-a="all">Run to end</button><button type="button" class="wb" data-a="r">Reset</button>');
  const draw = () => {
    const cur = path[Math.min(step, path.length - 1)], L = cur.L; let g = '';
    for (let i = 0; i < P.length; i++) if (lvl[i] >= L) nbr[L][i].forEach(j => { if (j > i || !nbr[L][j].includes(i)) g += `<line x1="${P[i][0]}" y1="${P[i][1]}" x2="${P[j][0]}" y2="${P[j][1]}" stroke="var(--line)" stroke-width="${L ? 1.4 : 0.8}"/>`; });
    P.forEach((p, i) => { if (lvl[i] >= L) g += `<circle cx="${p[0]}" cy="${p[1]}" r="${L ? 4.5 : 2.8}" fill="${i === truth ? 'var(--go)' : 'var(--line2)'}"/>`; });
    const vis = path.slice(0, step + 1); vis.forEach((s, k) => { if (k){ const a = P[vis[k - 1].n], b = P[s.n]; g += `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="var(--accent)" stroke-width="2.4"/>`; } });
    g += `<circle cx="${P[cur.n][0]}" cy="${P[cur.n][1]}" r="7" fill="var(--accent)"/><circle cx="${q[0]}" cy="${q[1]}" r="7" fill="none" stroke="var(--flame)" stroke-width="2.5"/><text x="${q[0] + 10}" y="${q[1] + 4}" style="fill:var(--flame);font-weight:600">query</text>`;
    fig.innerHTML = `<div class="wscroll"><svg class="wsvg" viewBox="0 0 600 360" style="min-width:420px;cursor:crosshair">${g}</svg></div>`;
    const svg = fig.querySelector('svg'); svg.onclick = e => { const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY; const p = pt.matrixTransform(svg.getScreenCTM().inverse()); q = [p.x, p.y]; truth = search(); step = 0; draw(); };
    const done = step >= path.length - 1;
    out.innerHTML = `<b>Layer ${L}</b> (${L === 2 ? 'top: few nodes, long links' : L === 1 ? 'middle' : 'bottom: every node, short links'}) · hop ${step + 1} of ${path.length}. ${done ? (cur.n === truth ? 'Reached the true nearest neighbour (green).' : 'Greedy search stopped at a local minimum, not the true nearest (green). Real HNSW keeps efSearch candidates to avoid this.') : 'Moves to whichever neighbour is closer to the query; drops a layer when none is.'} Click to move the query.`;
  };
  btns.onclick = e => { const a = e.target.closest('[data-a]')?.dataset.a; if (a === 's') step = Math.min(path.length - 1, step + 1); if (a === 'all') step = path.length - 1; if (a === 'r') step = 0; draw(); };
  root.append(btns, fig, out); draw();
};
WIDGETS.funnel = root => {
  const bar = WX.h('div'), out = WX.h('div', 'wout');
  const go = v => {
    const parts = [['Embed query', v.e, 'var(--muted)'], ['ANN (slowest shard)', v.a * (1 + Math.log2(v.sh) * 0.15), 'var(--accent)'], ['Merge', 1 + v.sh * 0.2, 'var(--line2)'], ['Rerank', v.rr * v.rp / 8, 'var(--flame)'], ['Fetch text', v.f, 'var(--go)']];
    const tot = parts.reduce((a, p) => a + p[1], 0);
    bar.innerHTML = `<div style="display:flex;height:26px;border-radius:7px;overflow:hidden">${parts.map(p => `<div title="${p[0]}: ${WX.f(p[1], 1)} ms" style="width:${p[1] / Math.max(tot, v.b) * 100}%;background:${p[2]}"></div>`).join('')}</div><div class="wnote" style="margin-top:6px">${parts.map(p => `<span style="display:inline-flex;align-items:center;gap:5px;margin-right:12px"><i style="width:9px;height:9px;border-radius:2px;background:${p[2]};display:inline-block"></i>${p[0]} ${WX.f(p[1], 1)} ms</span>`).join('')}</div>`;
    WX.out(out, [['Total retrieval latency', WX.f(tot, 0) + ' ms', 1], ['Budget', WX.f(v.b) + ' ms'], ['Status', tot <= v.b ? 'Within budget' : 'Over budget', tot > v.b]]);
  };
  const v = WX.ctl(root, [{k: 'e', l: 'Embedding call (ms)', min: 2, max: 150, v: 20}, {k: 'a', l: 'ANN per shard (ms)', min: 1, max: 100, v: 8}, {k: 'sh', l: 'Shards', min: 1, max: 64, v: 8}, {k: 'rr', l: 'Candidates reranked', min: 0, max: 300, step: 10, v: 100}, {k: 'rp', l: 'Cross-encoder ms per pair', min: 0.5, max: 10, step: 0.5, v: 2}, {k: 'f', l: 'Fetch text (ms)', min: 1, max: 50, v: 5}, {k: 'b', l: 'Latency budget (ms)', min: 50, max: 1000, step: 10, v: 150}], go);
  root.append(bar, out); go(v);
};
WIDGETS.segments = root => {
  let segs = [{id: 1, rows: 0, st: 'growing'}], nid = 2, t = 0;
  const CAP = 4000;
  const view = WX.h('div'), out = WX.h('div', 'wout'), log = WX.h('div', 'wlog', 'Insert data and watch segments seal and get indexed.');
  const btns = WX.h('div', 'wbtns', '<button type="button" class="wb" data-a="i">Insert 1,000 rows</button><button type="button" class="wb" data-a="t">Let 1 tick pass</button><button type="button" class="wb" data-a="r">Reset</button>');
  const tick = () => { t++; segs.forEach(s => { if (s.st === 'indexing' && t - s.at >= 2){ s.st = 'indexed'; log.textContent = `t=${t}: index built for segment ${s.id}. Queries now use ANN on it.\n` + log.textContent; } }); };
  const draw = () => {
    view.innerHTML = `<div style="display:flex;flex-wrap:wrap;gap:8px">${segs.map(s => `<div style="flex:0 0 120px;border-radius:9px;padding:9px;border:1.5px ${s.st === 'growing' ? 'dashed var(--flame)' : 'solid ' + (s.st === 'indexed' ? 'var(--go)' : 'var(--gold)')};background:var(--card)"><div style="font:600 12px var(--mono)">seg ${s.id}</div><div style="font-size:12px;color:var(--muted)">${s.st}</div><div class="bar" style="margin-top:6px;height:6px"><i style="width:${s.rows / CAP * 100}%;background:${s.st === 'growing' ? 'var(--flame)' : s.st === 'indexed' ? 'var(--go)' : 'var(--gold)'}"></i></div><div style="font:500 11px var(--mono);margin-top:4px">${WX.f(s.rows)} rows</div></div>`).join('')}</div>`;
    const brute = segs.filter(s => s.st !== 'indexed').reduce((a, s) => a + s.rows, 0), idx = segs.filter(s => s.st === 'indexed');
    WX.out(out, [['Rows searched by brute force', WX.f(brute), 1], ['Indexed segments', idx.length], ['Total rows', WX.f(segs.reduce((a, s) => a + s.rows, 0))]]);
  };
  btns.onclick = e => { const a = e.target.closest('[data-a]')?.dataset.a; if (!a) return;
    if (a === 'r'){ segs = [{id: 1, rows: 0, st: 'growing'}]; nid = 2; t = 0; log.textContent = 'Reset.'; }
    if (a === 't') tick();
    if (a === 'i'){ tick(); const g = segs.find(s => s.st === 'growing'); g.rows += 1000; log.textContent = `t=${t}: 1,000 rows written to the log, searchable in growing segment ${g.id}.\n` + log.textContent;
      if (g.rows >= CAP){ g.st = 'indexing'; g.at = t; segs.push({id: nid++, rows: 0, st: 'growing'}); log.textContent = `t=${t}: segment ${g.id} sealed and flushed; index build scheduled.\n` + log.textContent; } }
    draw(); };
  root.append(btns, view, out, log); draw();
};
// ---------- module 3 ----------
WIDGETS.nexttoken = root => {
  const T = {'is': [['Paris', 7.5], ['a', 4.2], ['the', 3.9], ['located', 3.1], ['known', 2.6], ['not', 1.2]], 'Paris': [['.', 6.0], [',', 5.1], ['and', 3.3], ['(', 2.0], ['which', 1.8]], 'a': [['city', 5.2], ['beautiful', 3.6], ['popular', 3.0], ['common', 2.2], ['trick', 1.0]], 'the': [['city', 4.9], ['answer', 3.4], ['Eiffel', 2.8], ['same', 1.9], ['capital', 1.5]], 'located': [['in', 6.5], ['on', 3.1], ['near', 2.9], ['along', 2.0]], 'known': [['as', 6.0], ['for', 5.4], ['to', 2.3]], ',': [['which', 4.4], ['the', 3.8], ['and', 3.6], ['a', 2.6]], '.': [['It', 5.0], ['The', 4.6], ['Paris', 3.0], ['\n', 2.5]], '_': [['the', 4.0], ['and', 3.8], ['of', 3.5], ['is', 3.0], ['a', 2.8], ['.', 2.4]]};
  let seq = ['The', 'capital', 'of', 'France', 'is'];
  const txt = WX.h('div', 'wtok'), rows = WX.h('div'), btns = WX.h('div', 'wbtns'), note = WX.h('div', 'wnote', 'Toy probabilities, hand-written to illustrate the loop. Click a candidate to append it; the model runs again on the longer sequence.');
  const draw = () => {
    const last = seq[seq.length - 1], cand = T[last] || T._, p = WX.softmax(cand.map(c => c[1]));
    txt.innerHTML = seq.map((t, i) => `<span style="background:${i >= 5 ? 'var(--accent-soft)' : 'var(--panel)'}">${esc(t === '\n' ? '↵' : t)}</span>`).join('');
    WX.rows(rows, cand.map((c, i) => ({l: c[0] === '\n' ? '↵' : c[0], v: p[i], max: 1, t: WX.pct(p[i]), hot: i === 0})));
    btns.innerHTML = cand.map((c, i) => `<button type="button" class="wb" data-i="${i}">+ ${esc(c[0] === '\n' ? '↵' : c[0])}</button>`).join('') + '<button type="button" class="wb" data-r="1">Reset</button>';
  };
  btns.onclick = e => { const b = e.target.closest('button'); if (!b) return; if (b.dataset.r) seq = ['The', 'capital', 'of', 'France', 'is']; else { const last = seq[seq.length - 1]; seq.push((T[last] || T._)[+b.dataset.i][0]); } draw(); };
  root.append(txt, rows, btns, note); draw();
};
WIDGETS.chinchilla = root => {
  const out = WX.h('div', 'wout'), ch = WX.h('div'), note = WX.h('div', 'wnote', 'Training compute C ≈ 6 × N × D FLOPs. Chinchilla-optimal: D ≈ 20 N. Inference costs about 2 N FLOPs per token, so a smaller model trained longer is cheaper to serve.');
  const go = v => {
    const C = 10 ** v.c, N = Math.sqrt(C / (6 * v.r)), D = v.r * N;
    WX.out(out, [['Parameters N', WX.big(N), 1], ['Training tokens D', WX.big(D), 1], ['Tokens per parameter', WX.f(v.r)], ['Inference FLOPs / token', WX.big(2 * N)]]);
    const pts = []; for (let r = 5; r <= 3000; r *= 1.15) pts.push([r, Math.sqrt(C / (6 * r))]);
    WX.line(ch, [{pts, l: 'Model size at this compute budget', dots: [[v.r, N]]}], {xlog: 1, ylog: 1, xl: 'Tokens per parameter (log)', yl: 'Parameters', marks: [{x: 20, l: 'Chinchilla'}]});
  };
  const v = WX.ctl(root, [{k: 'c', l: 'Compute budget (FLOPs)', min: 20, max: 26, step: 0.1, v: 23.8, fmt: x => '10^' + x.toFixed(1)}, {k: 'r', l: 'Tokens per parameter', min: 5, max: 3000, step: 5, v: 20}], go);
  root.append(out, ch, note); go(v);
};
WIDGETS.sampling = root => {
  const C = [['apples', 3.1], ['bananas', 2.6], ['mangoes', 2.2], ['strawberries', 1.9], ['grapes', 1.5], ['kiwi', 0.9], ['durian', 0.2], ['tomatoes', -0.4], ['broccoli', -1.5], ['rocks', -3.0]];
  const rows = WX.h('div'), out = WX.h('div', 'wout'), smp = WX.h('div', 'wtok'), btns = WX.h('div', 'wbtns', '<button type="button" class="wb" data-a="s">Sample 30 continuations</button>');
  let last = null;
  const dist = v => {
    if (v.t === 0){ return C.map((_, i) => i === 0 ? 1 : 0); }
    let p = WX.softmax(C.map(c => c[1]), v.t), keep = p.map(() => true);
    const ord = p.map((x, i) => [x, i]).sort((a, b) => b[0] - a[0]);
    ord.forEach(([, i], r) => { if (r >= v.k) keep[i] = false; });
    let cum = 0; ord.forEach(([x, i]) => { if (cum >= v.p) keep[i] = false; cum += x; });
    const mx = ord[0][0]; p.forEach((x, i) => { if (x < v.m * mx) keep[i] = false; });
    p = p.map((x, i) => keep[i] ? x : 0); const s = p.reduce((a, b) => a + b, 0); return p.map(x => x / s);
  };
  const go = v => { const p = dist(v); last = p; const H = -p.reduce((a, x) => a + (x > 0 ? x * Math.log2(x) : 0), 0);
    WX.rows(rows, C.map((c, i) => ({l: c[0], v: p[i], max: 1, t: p[i] ? WX.pct(p[i]) : 'cut', off: !p[i], hot: p[i] === Math.max(...p)})));
    WX.out(out, [['Tokens still possible', p.filter(x => x > 0).length, 1], ['Top token probability', WX.pct(Math.max(...p))], ['Entropy', WX.f(H, 2) + ' bits']]); };
  const v = WX.ctl(root, [{k: 't', l: 'Temperature', min: 0, max: 2, step: 0.05, v: 1}, {k: 'k', l: 'Top-k', min: 1, max: 10, v: 10}, {k: 'p', l: 'Top-p', min: 0.05, max: 1, step: 0.05, v: 1}, {k: 'm', l: 'Min-p', min: 0, max: 0.5, step: 0.01, v: 0}], go);
  btns.onclick = () => { const r = Math.random; smp.innerHTML = Array.from({length: 30}, () => { let u = r(), i = 0; while (i < last.length - 1 && u > last[i]){ u -= last[i]; i++; } return `<span style="background:var(--panel)">${C[i][0]}</span>`; }).join(''); };
  root.insertAdjacentHTML('afterbegin', '<div class="wnote">Prompt: “My favourite fruit is …”. The model\'s raw scores (logits) are fixed; only the decoding settings change.</div>');
  root.append(rows, out, btns, smp); go(v);
};
WIDGETS.lora = root => {
  const out = WX.h('div', 'wout');
  const go = v => { const full = v.L * v.mpl * v.d * v.d, lora = v.L * v.mpl * v.r * (v.d + v.d);
    WX.out(out, [['Weights in adapted matrices', WX.big(full)], ['LoRA trainable params', WX.big(lora), 1], ['Share trained', WX.pct(lora / full, 3), 1], ['Adapter file (fp16)', WX.bytes(lora * 2)], ['Full fine-tune optimizer state (Adam, fp32)', WX.bytes(full * 16)]]); };
  const v = WX.ctl(root, [{k: 'd', l: 'Hidden size d', min: 1024, max: 16384, step: 512, v: 4096}, {k: 'L', l: 'Layers', min: 8, max: 128, v: 32}, {k: 'mpl', l: 'Adapted matrices per layer', min: 1, max: 7, v: 4}, {k: 'r', l: 'LoRA rank r', min: 1, max: 256, v: 16}], go);
  root.append(out, WX.h('div', 'wnote', 'Each adapted d × d matrix W gets W + B·A, with A (r × d) and B (d × r). Only A and B train. Optimizer state for full fine-tuning is roughly 16 bytes per parameter (weights, gradients and two Adam moments in mixed precision).')); go(v);
};
WIDGETS.chunker = root => {
  const TXT = 'Acme Cloud bills storage monthly. Prices are set per region. In the EU region, standard storage costs 0.021 dollars per gigabyte. Archive storage is cheaper but has a retrieval fee. Retrieval from archive takes up to twelve hours. Data transfer between regions is billed separately. Transfers inside one region are free. Customers on the Enterprise plan get a 15 percent discount on storage. The discount does not apply to archive retrieval fees. Invoices are issued on the first business day of each month. Disputes must be raised within 30 days of the invoice date. After 30 days, charges are considered final. Support can extend this window only for billing errors caused by Acme.';
  const Qs = [['How much does EU standard storage cost for Enterprise customers?', [2, 7]], ['Can I dispute a charge after 45 days?', [10, 11, 12]]];
  const sents = TXT.match(/[^.]+\./g).map(s => s.trim());
  const view = WX.h('div'), out = WX.h('div', 'wout');
  const go = v => {
    const words = []; sents.forEach((s, si) => s.split(' ').forEach(w => words.push({w, si})));
    const chunks = []; const stepW = Math.max(1, v.size - v.ov); for (let i = 0; i < words.length; i += stepW){ chunks.push([i, Math.min(words.length, i + v.size)]); if (i + v.size >= words.length) break; }
    const need = Qs[v.q][1];
    const covers = chunks.map(([a, b]) => need.every(si => { const idx = words.map((w, k) => w.si === si ? k : -1).filter(k => k >= 0); return idx[0] >= a && idx[idx.length - 1] < b; }));
    const cols = ['var(--accent-soft)', 'var(--go-soft)', 'var(--gold-soft)', 'var(--flame-soft)'];
    view.innerHTML = `<div style="display:flex;flex-direction:column;gap:6px">${chunks.map(([a, b], ci) => `<div style="background:${cols[ci % 4]};border-radius:8px;padding:7px 10px;font-size:13px;border:1.5px solid ${covers[ci] ? 'var(--go)' : 'transparent'}"><span style="font:600 11px var(--mono);color:var(--muted)">chunk ${ci + 1} · ${b - a} words${covers[ci] ? ' · contains the whole answer' : ''}</span><br>${words.slice(a, b).map(w => need.includes(w.si) ? `<b style="color:var(--accent)">${esc(w.w)}</b>` : esc(w.w)).join(' ')}</div>`).join('')}</div>`;
    WX.out(out, [['Chunks', chunks.length], ['Chunks holding the full answer', covers.filter(Boolean).length, 1], ['Answer split across chunks', covers.some(Boolean) ? 'No' : 'Yes', !covers.some(Boolean)]]);
  };
  const v = WX.ctl(root, [{k: 'q', l: 'Question', opts: Qs.map((q, i) => [String(i), q[0]]), v: '0', num: 1}, {k: 'size', l: 'Chunk size (words)', min: 10, max: 120, step: 5, v: 25}, {k: 'ov', l: 'Overlap (words)', min: 0, max: 30, v: 0}], go);
  root.append(out, view); go(v);
};
// ---------- module 4 ----------
WIDGETS.rope = root => {
  const fig = WX.h('div'), out = WX.h('div', 'wout'), ch = WX.h('div');
  const go = v => {
    const th = v.f * Math.PI / 180, q0 = [1, 0.25], k0 = [0.8, 0.55];
    const rot = (p, a) => [p[0] * Math.cos(a) - p[1] * Math.sin(a), p[0] * Math.sin(a) + p[1] * Math.cos(a)];
    const q = rot(q0, v.m * th), k = rot(k0, v.n * th), dot = q[0] * k[0] + q[1] * k[1];
    const S = 110, o = 150, A = (p, c, l) => `<line x1="${o}" y1="${o}" x2="${o + p[0] * S}" y2="${o - p[1] * S}" stroke="${c}" stroke-width="3" marker-end="url(#ra)"/><text x="${o + p[0] * S * 1.12}" y="${o - p[1] * S * 1.12}" style="fill:${c};font-weight:600">${l}</text>`;
    fig.innerHTML = `<div class="wscroll"><svg class="wsvg" viewBox="0 0 300 300" style="max-width:300px;margin:0 auto"><defs><marker id="ra" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0L10,5L0,10z" fill="context-stroke"/></marker></defs><circle cx="${o}" cy="${o}" r="${S}" fill="none" stroke="var(--line)"/><line class="gr" x1="${o - S - 10}" x2="${o + S + 10}" y1="${o}" y2="${o}"/><line class="gr" y1="${o - S - 10}" y2="${o + S + 10}" x1="${o}" x2="${o}"/>${A(q0, 'var(--line2)', '')}${A(k0, 'var(--line2)', '')}${A(q, 'var(--accent)', 'q @ ' + v.m)}${A(k, 'var(--flame)', 'k @ ' + v.n)}</svg></div>`;
    WX.out(out, [['Query position m', v.m], ['Key position n', v.n], ['Offset m − n', v.m - v.n, 1], ['Score q·k', WX.f(dot, 4), 1]]);
    const pts = []; for (let d = -30; d <= 30; d++){ const a = rot(q0, d * th), b = k0; pts.push([d, a[0] * b[0] + a[1] * b[1]]); }
    WX.line(ch, [{pts, l: 'score as a function of offset (same for any absolute m, n)', dots: [[v.m - v.n, dot]]}], {xl: 'Offset m − n', yl: 'q·k', ymin: -1.2, ymax: 1.2, fx: x => WX.f(x), fy: y => WX.f(y, 1), xmin: -30, xmax: 30});
  };
  const v = WX.ctl(root, [{k: 'm', l: 'Query position m', min: 0, max: 60, v: 12}, {k: 'n', l: 'Key position n', min: 0, max: 60, v: 9}, {k: 'f', l: 'Rotation per position (degrees, one dim pair)', min: 1, max: 45, v: 12}], go);
  root.append(fig, out, ch, WX.h('div', 'wnote', 'Move m and n together: the score does not change, because only the angle between the rotated vectors matters. Real RoPE applies this to every pair of dimensions, each pair at a different speed.')); go(v);
};
WIDGETS.attention = root => {
  const T = ['The', 'animal', "didn't", 'cross', 'the', 'street', 'because', 'it', 'was', 'too', 'tired'];
  const S = T.map((_, i) => T.map((_, j) => 0.2 * Math.exp(-Math.abs(i - j) / 2)));
  const set = (i, j, x) => { S[i][j] += x; };
  set(7, 1, 3.4); set(7, 5, 1.6); set(7, 0, 0.6); set(10, 1, 2.2); set(10, 7, 2.6); set(10, 8, 1.4); set(3, 1, 1.8); set(3, 5, 1.6); set(5, 4, 1.6); set(1, 0, 1.5); set(9, 10, 0.5); set(8, 7, 1.9); set(6, 3, 1.1);
  let sel = 7;
  const grid = WX.h('div'), rows = WX.h('div');
  const go = v => {
    const n = T.length, W = T.map((_, i) => { const z = S[i].map((x, j) => (v.c && j > i) ? -Infinity : x * v.s); const m = Math.max(...z.filter(isFinite)); const e = z.map(x => isFinite(x) ? Math.exp(x - m) : 0); const s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); });
    let h = `<div class="wscroll"><div class="wgrid" style="grid-template-columns:62px repeat(${n},minmax(22px,1fr));min-width:420px;font:500 10.5px var(--mono);color:var(--muted)"><span></span>${T.map(t => `<span style="writing-mode:vertical-rl;transform:rotate(180deg);height:52px;overflow:hidden">${t}</span>`).join('')}`;
    W.forEach((r, i) => { h += `<span data-r="${i}" style="text-align:right;padding-right:5px;cursor:pointer;${i === sel ? 'color:var(--accent);font-weight:700' : ''}">${T[i]}</span>` + r.map((x, j) => `<i data-r="${i}" title="${T[i]} → ${T[j]}: ${WX.pct(x)}" style="cursor:pointer;background:${v.c && j > i ? 'transparent;border:1px dashed var(--line)' : `color-mix(in srgb,var(--accent) ${Math.round(Math.min(1, x * 1.6) * 100)}%,var(--panel))`};${i === sel ? 'outline:1.5px solid var(--accent)' : ''}"></i>`).join(''); });
    grid.innerHTML = h + '</div></div>';
    WX.rows(rows, T.map((t, j) => ({l: t, v: W[sel][j], max: 1, t: WX.pct(W[sel][j]), off: v.c && j > sel, hot: W[sel][j] === Math.max(...W[sel])})));
    grid.querySelectorAll('[data-r]').forEach(e => e.onclick = () => { sel = +e.dataset.r; go(v); });
  };
  const v = WX.ctl(root, [{k: 's', l: 'Score sharpness (scale)', min: 0.2, max: 3, step: 0.1, v: 1}, {k: 'c', l: 'Causal mask', opts: [['0', 'Off (encoder)'], ['1', 'On (decoder)']], v: '0', num: 1}], go);
  root.append(grid, WX.h('div', 'wnote', 'Click a row (a query token) to see where it puts its attention. Scores are hand-set for illustration, not taken from a real model.'), rows); go(v);
};
WIDGETS.scaling = root => {
  const out = WX.h('div', 'wout'), rows = WX.h('div');
  const go = v => {
    const R = WX.rng(v.seed + 1), n = 8;
    const q = Array.from({length: v.d}, () => WX.gauss(R)), ks = Array.from({length: n}, () => Array.from({length: v.d}, () => WX.gauss(R)));
    const sc = ks.map(k => k.reduce((a, x, i) => a + x * q[i], 0)), use = v.sc ? sc.map(x => x / Math.sqrt(v.d)) : sc, p = WX.softmax(use);
    const sd = Math.sqrt(sc.reduce((a, x) => a + x * x, 0) / n);
    WX.rows(rows, p.map((x, i) => ({l: 'key ' + (i + 1), v: x, max: 1, t: WX.pct(x), hot: x === Math.max(...p)})));
    WX.out(out, [['Spread of raw scores (std)', WX.f(sd, 1), 1], ['Expected √d_k', WX.f(Math.sqrt(v.d), 1)], ['Largest weight', WX.pct(Math.max(...p)), 1], ['Softmax state', Math.max(...p) > 0.95 ? 'Saturated (near one-hot)' : 'Smooth']]);
  };
  const v = WX.ctl(root, [{k: 'd', l: 'Key dimension d_k', min: 2, max: 512, v: 128}, {k: 'sc', l: 'Divide by √d_k', opts: [['1', 'Yes'], ['0', 'No']], v: '1', num: 1}, {k: 'seed', l: 'Random draw', min: 1, max: 20, v: 1}], go);
  root.append(out, rows, WX.h('div', 'wnote', 'Random unit-variance queries and keys. Without scaling, score spread grows with √d_k and softmax collapses onto one key, which also kills the gradients for the others.')); go(v);
};
WIDGETS.params = root => {
  const rows = WX.h('div'), out = WX.h('div', 'wout');
  const go = v => {
    const d = v.d, kvd = d * v.kv / v.h, att = v.L * (d * d * 2 + d * kvd * 2), ff = v.L * 3 * d * v.ff * d, emb = v.V * d * (v.tie ? 1 : 2), tot = att + ff + emb;
    WX.rows(rows, [{l: 'Attention', v: att, max: tot, t: WX.big(att)}, {l: 'MLP (SwiGLU)', v: ff, max: tot, t: WX.big(ff), hot: 1}, {l: 'Embeddings', v: emb, max: tot, t: WX.big(emb)}]);
    WX.out(out, [['Total parameters', WX.big(tot), 1], ['fp16 weights', WX.bytes(tot * 2)], ['MLP share', WX.pct(ff / tot, 0)], ['Per layer', WX.big((att + ff) / v.L)]]);
  };
  const v = WX.ctl(root, [{k: 'd', l: 'Hidden size d_model', min: 512, max: 16384, step: 256, v: 4096}, {k: 'L', l: 'Layers', min: 2, max: 128, v: 32}, {k: 'h', l: 'Query heads', min: 4, max: 128, step: 4, v: 32}, {k: 'kv', l: 'KV heads', min: 1, max: 128, v: 8}, {k: 'ff', l: 'MLP width ÷ d_model', min: 2, max: 6, step: 0.25, v: 3.5}, {k: 'V', l: 'Vocabulary', min: 32000, max: 256000, step: 1000, v: 128000}, {k: 'tie', l: 'Tied input/output embeddings', opts: [['0', 'No'], ['1', 'Yes']], v: '0', num: 1}], go);
  root.append(rows, out, WX.h('div', 'wnote', 'Defaults are close to Llama 3 8B (32 layers, d = 4,096, 8 KV heads, MLP 14,336, 128k vocab) and give about 8B. Norm weights and biases are ignored.')); go(v);
};
WIDGETS.kvcache = root => {
  const bar = WX.h('div'), out = WX.h('div', 'wout'), note = WX.h('div', 'wnote');
  const go = v => {
    const per = 2 * v.L * v.kv * v.hd * v.b, kv = per * v.ctx * v.bs, w = v.p * 1e9 * v.wb, gpu = v.gpu * 1024 ** 3;
    const tot = kv + w;
    bar.innerHTML = `<div style="position:relative;height:30px;border-radius:7px;overflow:hidden;background:var(--panel)"><div style="position:absolute;left:0;top:0;bottom:0;width:${Math.min(100, w / Math.max(tot, gpu) * 100)}%;background:var(--muted)"></div><div style="position:absolute;top:0;bottom:0;left:${Math.min(100, w / Math.max(tot, gpu) * 100)}%;width:${Math.min(100, kv / Math.max(tot, gpu) * 100)}%;background:var(--accent)"></div><div style="position:absolute;top:-2px;bottom:-2px;left:${gpu / Math.max(tot, gpu) * 100}%;border-left:2px dashed var(--bad)"></div></div><div class="wnote" style="margin-top:6px"><span style="color:var(--muted)">■</span> weights &nbsp; <span style="color:var(--accent)">■</span> KV cache &nbsp; <span style="color:var(--bad)">┊</span> GPU memory</div>`;
    const maxB = Math.floor(Math.max(0, gpu * 0.9 - w) / (per * v.ctx));
    WX.out(out, [['KV per token', WX.bytes(per), 1], ['KV per sequence', WX.bytes(per * v.ctx)], ['KV for the batch', WX.bytes(kv), 1], ['Weights', WX.bytes(w)], ['Max batch that fits (90%)', WX.f(maxB), 1]]);
    note.textContent = tot > gpu ? 'Does not fit. Reduce batch or context, quantize the cache, or shard across GPUs.' : kv > w ? 'The KV cache is now bigger than the model weights.' : 'Weights dominate memory at this batch and context.';
  };
  const v = WX.ctl(root, [{k: 'L', l: 'Layers', min: 8, max: 128, v: 32}, {k: 'kv', l: 'KV heads', min: 1, max: 64, v: 8}, {k: 'hd', l: 'Head dim', opts: [['64', '64'], ['128', '128'], ['256', '256']], v: '128', num: 1}, {k: 'b', l: 'KV precision', opts: [['2', 'fp16 / bf16'], ['1', 'fp8 / int8'], ['0.5', 'int4']], v: '2', num: 1}, {k: 'ctx', l: 'Context tokens', min: 512, max: 131072, step: 512, v: 8192}, {k: 'bs', l: 'Batch (sequences)', min: 1, max: 256, v: 16}, {k: 'p', l: 'Model params (B)', min: 1, max: 405, v: 8}, {k: 'wb', l: 'Weight precision', opts: [['2', 'fp16'], ['1', 'int8 / fp8'], ['0.5', '4-bit']], v: '2', num: 1}, {k: 'gpu', l: 'GPU memory (GB)', opts: [['24', '24'], ['48', '48'], ['80', '80'], ['141', '141'], ['192', '192']], v: '80', num: 1}], go);
  root.append(bar, out, note); go(v);
};
WIDGETS.complexity = root => {
  const ch = WX.h('div'), out = WX.h('div', 'wout');
  const go = v => {
    const n = v.n, pts1 = [], pts2 = [];
    for (let x = 100; x <= 131072; x *= 1.3){ pts1.push([x, x]); pts2.push([x, x * x / 2]); }
    WX.line(ch, [{pts: pts1, l: 'RNN: sequential steps (n)', c: 'var(--flame)'}, {pts: pts2, l: 'Causal attention: token pairs scored (n²/2)'}], {xlog: 1, ylog: 1, xl: 'Sequence length (log)', yl: 'Count (log)', marks: [{x: n, l: WX.big(n)}]});
    WX.out(out, [['RNN steps that cannot run in parallel', WX.big(n)], ['Attention pairs per head per layer', WX.big(n * n / 2), 1], ['RNN state at inference', 'constant'], ['Transformer KV entries', WX.big(n) + ' per layer', 1]]);
  };
  const v = WX.ctl(root, [{k: 'n', l: 'Sequence length', min: 128, max: 131072, step: 128, v: 8192}], go);
  root.append(ch, out); go(v);
};
// ---------- module 5 ----------
WIDGETS.mask = root => {
  const n = 10, grid = WX.h('div'), out = WX.h('div', 'wnote');
  const go = v => {
    const ok = (i, j) => v.t === 'causal' ? j <= i : v.t === 'bi' ? true : v.t === 'slide' ? j <= i && i - j < v.w : v.t === 'doc' ? j <= i && ((i < 4 && j < 4) || (i >= 4 && i < 7 && j >= 4 && j < 7) || (i >= 7 && j >= 7)) : j <= i && j < 7 && i < 7;
    let h = `<div class="wscroll"><div class="wgrid" style="grid-template-columns:44px repeat(${n},minmax(22px,34px));min-width:360px;font:500 10.5px var(--mono);color:var(--muted)"><span></span>${Array.from({length: n}, (_, j) => `<span style="text-align:center">k${j}</span>`).join('')}`;
    for (let i = 0; i < n; i++){ h += `<span style="text-align:right;padding-right:5px">q${i}</span>`; for (let j = 0; j < n; j++){ const a = ok(i, j); h += `<i title="q${i} ${a ? 'can' : 'cannot'} see k${j}" style="background:${a ? 'var(--accent)' : 'var(--panel)'};${a ? '' : 'border:1px dashed var(--line)'}"></i>`; } }
    grid.innerHTML = h + '</div></div>';
    const cnt = Array.from({length: n}, (_, i) => Array.from({length: n}, (_, j) => ok(i, j) ? 1 : 0).reduce((a, b) => a + b, 0)).reduce((a, b) => a + b, 0);
    out.innerHTML = `Filled cells are allowed (score kept); empty cells get −∞ before softmax. ${cnt} of ${n * n} pairs computed. ` + ({causal: 'Lower triangle: each token sees itself and the past.', bi: 'Every token sees every token. Used by encoders such as BERT.', slide: `Each token sees only the last ${v.w} tokens.`, doc: 'Three packed documents (tokens 0-3, 4-6, 7-9): causal inside each, no attention across.', pad: 'Tokens 7-9 are padding: nobody attends to them and their rows are ignored.'})[v.t];
  };
  const v = WX.ctl(root, [{k: 't', l: 'Mask type', opts: [['causal', 'Causal (decoder)'], ['bi', 'Bidirectional (encoder)'], ['slide', 'Sliding window'], ['doc', 'Packed documents'], ['pad', 'Causal + padding']], v: 'causal'}, {k: 'w', l: 'Window (sliding only)', min: 1, max: 10, v: 3}], go);
  root.append(grid, out); go(v);
};
WIDGETS.heads = root => {
  const bar = WX.h('div'), out = WX.h('div', 'wout');
  const go = v => {
    const kvh = v.t === 'mha' ? v.h : v.t === 'mqa' ? 1 : v.g, hd = v.d / v.h, grp = v.h / kvh;
    const cols = ['var(--accent)', 'var(--go)', 'var(--gold)', 'var(--flame)', 'var(--muted)'];
    const shown = Math.min(v.h, 64);
    bar.innerHTML = `<div class="wnote" style="margin-bottom:4px">Query heads (${v.h} × ${WX.f(hd, 1)} dims), coloured by the K/V head they share:</div><div style="display:flex;gap:2px">${Array.from({length: shown}, (_, i) => `<i style="flex:1;height:20px;border-radius:2px;background:${cols[Math.floor(i / grp) % 5]}"></i>`).join('')}</div><div class="wnote" style="margin:8px 0 4px">K/V heads (${kvh}):</div><div style="display:flex;gap:2px;width:${Math.max(4, kvh / v.h * 100)}%">${Array.from({length: Math.min(kvh, 64)}, (_, i) => `<i style="flex:1;height:20px;border-radius:2px;background:${cols[i % 5]}"></i>`).join('')}</div>`;
    const kvTok = 2 * v.L * kvh * hd * 2;
    WX.out(out, [['Head dimension', Number.isInteger(hd) ? hd : WX.f(hd, 2) + ' (must divide)'], ['Query heads per K/V head', WX.f(grp, 2)], ['KV cache per token (fp16)', WX.bytes(kvTok), 1], ['vs full multi-head', WX.pct(kvh / v.h, 1), 1]]);
  };
  const v = WX.ctl(root, [{k: 'd', l: 'd_model', min: 512, max: 8192, step: 512, v: 4096}, {k: 'h', l: 'Query heads', opts: [['8', '8'], ['16', '16'], ['32', '32'], ['64', '64']], v: '32', num: 1}, {k: 't', l: 'Variant', opts: [['mha', 'Multi-head (MHA)'], ['gqa', 'Grouped-query (GQA)'], ['mqa', 'Multi-query (MQA)']], v: 'gqa'}, {k: 'g', l: 'KV groups (GQA)', opts: [['2', '2'], ['4', '4'], ['8', '8']], v: '8', num: 1}, {k: 'L', l: 'Layers', min: 8, max: 96, v: 32}], go);
  root.append(bar, out); go(v);
};
WIDGETS.shapes = root => {
  const tbl = WX.h('div');
  const go = v => {
    const hd = v.d / v.h, R = [
      ['Token ids', `(${v.B}, ${v.T})`, v.B * v.T], ['Embeddings', `(${v.B}, ${v.T}, ${v.d})`, v.B * v.T * v.d], ['Q after head split', `(${v.B}, ${v.h}, ${v.T}, ${hd})`, v.B * v.h * v.T * hd],
      ['Scores Q·Kᵀ', `(${v.B}, ${v.h}, ${v.T}, ${v.T})`, v.B * v.h * v.T * v.T, 1], ['Attention output', `(${v.B}, ${v.h}, ${v.T}, ${hd})`, v.B * v.h * v.T * hd], ['Merged + W_O', `(${v.B}, ${v.T}, ${v.d})`, v.B * v.T * v.d],
      ['MLP hidden', `(${v.B}, ${v.T}, ${v.ff})`, v.B * v.T * v.ff], ['Logits', `(${v.B}, ${v.T}, ${v.V})`, v.B * v.T * v.V, 1]];
    tbl.innerHTML = `<div class="wscroll"><table class="wtbl"><tr><th>Stage</th><th>Shape</th><th>Elements</th><th>fp16 size</th></tr>${R.map(r => `<tr class="${r[3] ? 'hot' : ''}"><td>${r[0]}</td><td class="n">${r[1]}</td><td class="n">${WX.big(r[2])}</td><td class="n">${WX.bytes(r[2] * 2)}</td></tr>`).join('')}</table></div><div class="wnote">Highlighted rows grow fastest: scores with T², logits with vocabulary. ${Number.isInteger(hd) ? '' : 'Heads must divide d_model.'}</div>`;
  };
  const v = WX.ctl(root, [{k: 'B', l: 'Batch B', min: 1, max: 64, v: 2}, {k: 'T', l: 'Sequence T', min: 8, max: 32768, step: 8, v: 10}, {k: 'd', l: 'd_model', min: 64, max: 8192, step: 64, v: 512}, {k: 'h', l: 'Heads', opts: [['4', '4'], ['8', '8'], ['16', '16'], ['32', '32']], v: '8', num: 1}, {k: 'ff', l: 'MLP width', min: 256, max: 32768, step: 256, v: 2048}, {k: 'V', l: 'Vocabulary', min: 1000, max: 256000, step: 1000, v: 32000}], go);
  root.append(tbl); go(v);
};
// ---------- module 6 ----------
WIDGETS.paging = root => {
  const NB = 48, BS = 4, MAXL = 32; let reqs = [], free = Array.from({length: NB}, (_, i) => i), nid = 1, R = WX.rng(4);
  const cols = ['#5A4FCF', '#1F7A72', '#C99A3A', '#C2577A', '#6D8FD6', '#7A8B3A', '#B5651D', '#3A8FA0', '#8E5BB5', '#4C7A3F'];
  const grid = WX.h('div'), out = WX.h('div', 'wout'), log = WX.h('div', 'wnote');
  const btns = WX.h('div', 'wbtns', '<button type="button" class="wb" data-a="add">+ New request</button><button type="button" class="wb" data-a="step">Decode 1 token (all)</button><button type="button" class="wb" data-a="step4">Decode 4 tokens</button><button type="button" class="wb" data-a="fin">Finish oldest</button><button type="button" class="wb" data-a="r">Reset</button>');
  const need = r => Math.ceil(r.len / BS);
  const grow = r => { while (r.blocks.length < need(r)){ if (!free.length) return false; r.blocks.push(free.shift()); } return true; };
  const draw = msg => {
    const owner = {}; reqs.forEach(r => r.blocks.forEach((b, k) => owner[b] = {r, k}));
    grid.innerHTML = `<div class="wgrid" style="grid-template-columns:repeat(12,minmax(20px,1fr))">${Array.from({length: NB}, (_, b) => { const o = owner[b]; if (!o) return '<i style="background:var(--panel);border:1px dashed var(--line)"></i>'; const fill = Math.min(BS, o.r.len - o.k * BS); return `<i title="request ${o.r.id}, logical block ${o.k}, ${fill}/${BS} slots" style="background:linear-gradient(0deg,${cols[o.r.id % cols.length]} ${fill / BS * 100}%,color-mix(in srgb,${cols[o.r.id % cols.length]} 30%,var(--card)) 0)"></i>`; }).join('')}</div>`;
    const used = reqs.reduce((a, r) => a + r.len, 0), alloc = reqs.reduce((a, r) => a + r.blocks.length * BS, 0), contig = Math.floor(NB * BS / MAXL);
    WX.out(out, [['Active requests', reqs.length, 1], ['Blocks used', `${NB - free.length} / ${NB}`], ['Slots holding tokens', alloc ? WX.pct(used / alloc, 0) : '—', 1], ['Contiguous max-length allocator would fit', contig + ' requests']]);
    log.innerHTML = (msg || 'Each square is a physical block of 4 token slots. A request\'s blocks can be anywhere; its block table maps them in order.') + ' ' + reqs.map(r => `<span style="color:${cols[r.id % cols.length]};font-weight:600">#${r.id}</span>: [${r.blocks.join(', ')}]`).join(' &nbsp; ');
  };
  btns.onclick = e => { const a = e.target.closest('[data-a]')?.dataset.a; if (!a) return; let msg = '';
    if (a === 'r'){ reqs = []; free = Array.from({length: NB}, (_, i) => i); nid = 1; }
    if (a === 'add'){ const r = {id: nid++, len: 2 + Math.floor(R() * 9), blocks: []}; if (grow(r)) reqs.push(r); else { r.blocks.forEach(b => free.push(b)); msg = 'Out of blocks: a real scheduler would queue or preempt this request.'; } }
    if (a === 'step' || a === 'step4'){ for (let s = 0; s < (a === 'step' ? 1 : 4); s++) reqs.forEach(r => { if (r.len < MAXL){ r.len++; if (!grow(r)){ r.len--; msg = 'A request needed a new block but none were free.'; } } }); }
    if (a === 'fin' && reqs.length){ const r = reqs.shift(); free.push(...r.blocks); free.sort((x, y) => x - y); msg = `Request ${r.id} finished; its ${r.blocks.length} blocks went back to the free list.`; }
    draw(msg); };
  root.append(btns, grid, out, log); draw();
};
WIDGETS.pagecalc = root => {
  const out = WX.h('div', 'wout'), rows = WX.h('div');
  const go = v => {
    const per = 2 * v.L * v.kv * v.hd * 2, blk = per * v.bs, nb = Math.floor(v.free * 1024 ** 3 / blk), need = Math.ceil(v.len / v.bs), paged = Math.floor(nb / need);
    const contig = Math.floor(v.free * 1024 ** 3 / (per * v.max)), pre = Math.ceil(v.pre / v.bs), own = Math.ceil((v.len - v.pre) / v.bs), shared = v.pre > 0 ? Math.floor((nb - pre) / own) : paged;
    WX.out(out, [['KV per token', WX.bytes(per)], ['Block size', WX.bytes(blk)], ['Blocks available', WX.f(nb)], ['Blocks per request', need]]);
    WX.rows(rows, [{l: 'Contiguous', v: contig, t: WX.f(contig)}, {l: 'Paged', v: paged, t: WX.f(paged), hot: 1}, {l: 'Paged + shared prefix', v: shared, t: WX.f(shared), hot: 1}]);
  };
  const v = WX.ctl(root, [{k: 'free', l: 'Free GPU memory for KV (GB)', min: 4, max: 160, v: 24}, {k: 'L', l: 'Layers', min: 8, max: 128, v: 32}, {k: 'kv', l: 'KV heads', min: 1, max: 64, v: 8}, {k: 'hd', l: 'Head dim', min: 64, max: 256, step: 64, v: 128}, {k: 'bs', l: 'Block size (tokens)', opts: [['8', '8'], ['16', '16'], ['32', '32']], v: '16', num: 1}, {k: 'len', l: 'Actual tokens per request', min: 64, max: 32768, step: 64, v: 1000}, {k: 'max', l: 'Max length (contiguous reserve)', min: 512, max: 131072, step: 512, v: 4096}, {k: 'pre', l: 'Shared prefix tokens', min: 0, max: 4096, step: 16, v: 512}], go);
  root.append(out, WX.h('div', 'wnote', 'Concurrent requests that fit:'), rows); go(v);
};
WIDGETS.moe = root => {
  const toks = ['The', 'integral', 'of', 'x²', 'is', 'def', 'return', '{', 'Paris', 'bonjour'];
  const R = WX.rng(21), E = 8, L = toks.map(() => Array.from({length: E}, () => WX.gauss(R) * 1.4));
  let sel = 1;
  const chips = WX.h('div', 'wchips'), rows = WX.h('div'), out = WX.h('div', 'wout'), load = WX.h('div');
  const go = v => {
    chips.innerHTML = toks.map((t, i) => `<button type="button" class="wchip${i === sel ? ' on' : ''}" data-i="${i}">${esc(t)}</button>`).join('');
    const p = WX.softmax(L[sel]), ord = p.map((x, i) => [x, i]).sort((a, b) => b[0] - a[0]), top = new Set(ord.slice(0, v.k).map(x => x[1])), s = ord.slice(0, v.k).reduce((a, x) => a + x[0], 0);
    WX.rows(rows, p.map((x, i) => ({l: 'Expert ' + (i + 1), v: x, max: 1, t: top.has(i) ? 'w ' + WX.f(x / s, 2) : WX.pct(x), off: !top.has(i), hot: top.has(i)})));
    const cnt = Array(E).fill(0); L.forEach(l => { const q = WX.softmax(l).map((x, i) => [x, i]).sort((a, b) => b[0] - a[0]).slice(0, v.k); q.forEach(([, i]) => cnt[i]++); });
    load.innerHTML = '<div class="wnote" style="margin-bottom:4px">Load across all 10 tokens (balanced would be equal bars):</div>'; const lb = WX.h('div'); WX.rows(lb, cnt.map((c, i) => ({l: 'Expert ' + (i + 1), v: c, t: c + ' tok'}))); load.appendChild(lb);
    const shared = v.P * 0.2, exp = v.P * 0.8 / E;
    WX.out(out, [['Total parameters', WX.f(v.P, 1) + 'B'], ['Active per token', WX.f(shared + exp * v.k, 1) + 'B', 1], ['Active share', WX.pct((shared + exp * v.k) / v.P, 0), 1]]);
    chips.querySelectorAll('[data-i]').forEach(b => b.onclick = () => { sel = +b.dataset.i; go(v); });
  };
  const v = WX.ctl(root, [{k: 'k', l: 'Experts per token (top-k)', min: 1, max: 4, v: 2}, {k: 'P', l: 'Total model size (B params)', min: 8, max: 700, v: 47}], go);
  root.append(chips, rows, out, load, WX.h('div', 'wnote', 'Router scores are random here; a trained router learns them. The parameter split assumes about 20% shared (attention, embeddings) and 80% in 8 experts.')); go(v);
};
WIDGETS.flash = root => {
  const fig = WX.h('div'), out = WX.h('div', 'wout'), st = WX.h('div', 'wnote');
  let step = -1;
  const btns = WX.h('div', 'wbtns', '<button type="button" class="wb" data-a="s">Next tile ›</button><button type="button" class="wb" data-a="r">Reset</button>');
  const go = v => {
    const nb = Math.ceil(8 / v.b), total = nb * nb, cur = step < 0 ? null : [Math.floor(step / nb), step % nb];
    let g = ''; const c = 30;
    for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++){ const bi = Math.floor(i / v.b), bj = Math.floor(j / v.b), id = bi * nb + bj, on = cur && bi === cur[0] && bj === cur[1], done = step >= 0 && id < step;
      g += `<rect x="${60 + j * c}" y="${30 + i * c}" width="${c - 2}" height="${c - 2}" rx="3" fill="${on ? 'var(--accent)' : done ? 'color-mix(in srgb,var(--accent) 25%,var(--card))' : 'var(--panel)'}"/>`; }
    g += `<text x="${60 + 4 * c}" y="20" text-anchor="middle">keys / values (blocks)</text><text x="40" y="${30 + 4 * c}" text-anchor="middle" transform="rotate(-90 40 ${30 + 4 * c})">queries (blocks)</text>`;
    fig.innerHTML = `<div class="wscroll"><svg class="wsvg" viewBox="0 0 330 280" style="max-width:340px;margin:0 auto">${g}</svg></div>`;
    st.textContent = step < 0 ? 'The full 8 × 8 score matrix is never stored. Step through the tiles that pass through on-chip SRAM.' : `Tile ${step + 1} of ${total}: load Q block ${cur[0] + 1} and K/V block ${cur[1] + 1} into SRAM, compute scores, update the running max and sum, add to the partial output.${cur[1] === nb - 1 ? ' Last K/V block for this row: rescale and write this output block to HBM.' : ''}`;
    const n = 10 ** v.T, d = 128, by = 2, std = (n * n * 2 * 2 + 4 * n * d) * by, fl = 4 * n * d * by * Math.ceil(n / (v.b * 16));
    WX.out(out, [['Score matrix if stored (per head)', WX.bytes(n * n * by), 1], ['HBM traffic, standard (approx.)', WX.bytes(std)], ['HBM traffic, tiled (approx.)', WX.bytes(Math.min(fl, std)), 1], ['Extra memory, tiled', 'O(n), no n × n buffer']]);
  };
  btns.onclick = e => { const a = e.target.closest('[data-a]')?.dataset.a; const nb = Math.ceil(8 / v.b); if (a === 's') step = step + 1 >= nb * nb ? -1 : step + 1; if (a === 'r') step = -1; go(v); };
  const v = WX.ctl(root, [{k: 'b', l: 'Tile size (rows per block)', opts: [['2', '2'], ['4', '4']], v: '2', num: 1}, {k: 'T', l: 'Sequence length for the traffic estimate', min: 3, max: 5.3, step: 0.1, v: 4.5, fmt: x => WX.big(10 ** x) + ' tokens'}], v2 => { step = -1; go(v2); });
  root.append(btns, fig, st, out, WX.h('div', 'wnote', 'Traffic numbers are rough: standard attention writes and re-reads the n × n scores and probabilities; tiled attention re-reads K and V once per query block.')); go(v);
};
// ---------- module 7 ----------
WIDGETS.quant = root => {
  const ch = WX.h('div'), out = WX.h('div', 'wout');
  const go = v => {
    const R = WX.rng(5), n = v.blk, w = Array.from({length: n}, () => WX.gauss(R) * 0.02);
    if (v.o) w[3] = 0.02 * v.o;
    const levels = 2 ** (v.bits - 1) - 1, mx = Math.max(...w.map(Math.abs)), sc = levels / mx, q = w.map(x => Math.round(x * sc) / sc);
    const err = Math.sqrt(w.reduce((a, x, i) => a + (x - q[i]) ** 2, 0) / n), rel = err / Math.sqrt(w.reduce((a, x) => a + x * x, 0) / n);
    const shown = Math.min(n, 40), W2 = 560, H = 180, bw = W2 / shown, Y = x => H / 2 - x / (mx * 1.1) * (H / 2 - 8);
    let g = `<line class="ax" x1="0" x2="${W2}" y1="${H / 2}" y2="${H / 2}"/>`;
    if (levels <= 15) for (let l = -levels; l <= levels; l++) g += `<line x1="0" x2="${W2}" y1="${Y(l / sc)}" y2="${Y(l / sc)}" stroke="var(--line)" stroke-dasharray="2 4"/>`;
    for (let i = 0; i < shown; i++){ g += `<rect x="${i * bw + 2}" y="${Math.min(Y(w[i]), H / 2)}" width="${bw / 2 - 2}" height="${Math.abs(Y(w[i]) - H / 2)}" fill="var(--line2)"/><rect x="${i * bw + bw / 2}" y="${Math.min(Y(q[i]), H / 2)}" width="${bw / 2 - 2}" height="${Math.abs(Y(q[i]) - H / 2)}" fill="var(--accent)"/>`; }
    ch.innerHTML = `<div class="wscroll"><svg class="wsvg" viewBox="0 0 ${W2} ${H}" style="min-width:420px">${g}</svg></div><div class="wnote"><span style="color:var(--line2)">■</span> original weight &nbsp; <span style="color:var(--accent)">■</span> after ${v.bits}-bit absmax round-trip${levels <= 15 ? ' · dashed lines are the allowed levels' : ''}</div>`;
    const zero = q.filter(x => x === 0).length;
    WX.out(out, [['Levels each side', levels], ['RMS error / RMS weight', WX.pct(rel, 1), 1], ['Weights rounded to 0', `${zero} of ${n}`, zero > n / 3], ['70B model weights', WX.bytes(70e9 * v.bits / 8), 1]]);
  };
  const v = WX.ctl(root, [{k: 'bits', l: 'Bits per weight', opts: [['8', '8 (int8)'], ['6', '6'], ['4', '4 (int4)'], ['3', '3'], ['2', '2']], v: '4', num: 1}, {k: 'o', l: 'Outlier size (× typical)', min: 0, max: 30, v: 0, fmt: x => x ? x + '×' : 'none'}, {k: 'blk', l: 'Block size sharing one scale', opts: [['32', '32'], ['64', '64'], ['128', '128'], ['1024', '1024']], v: '64', num: 1}], go);
  root.append(ch, out, WX.h('div', 'wnote', 'Add an outlier: one big weight stretches the scale, and the small weights collapse onto a few levels or to zero. That is why per-block scales, GPTQ and AWQ exist.')); go(v);
};
WIDGETS.sparse = root => {
  const n = 24, grid = WX.h('div'), out = WX.h('div', 'wout');
  const go = v => {
    const ok = (i, j) => { if (j > i) return false; if (v.t === 'full') return true; const loc = i - j < v.w; if (v.t === 'window') return loc; if (v.t === 'strided') return loc || (i - j) % v.w === 0; if (v.t === 'global') return loc || j < 2 || i < 2; return false; };
    let h = `<div class="wscroll"><div class="wgrid" style="grid-template-columns:repeat(${n},minmax(10px,1fr));min-width:300px;max-width:480px">`; let c = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++){ const a = ok(i, j); if (a) c++; h += `<i style="background:${a ? (j < 2 && v.t === 'global' ? 'var(--flame)' : 'var(--accent)') : 'var(--panel)'}"></i>`; }
    grid.innerHTML = h + '</div></div>';
    const full = n * (n + 1) / 2, reach = v.t === 'full' ? 'whole context' : v.t === 'window' ? WX.f(v.L * (v.w - 1)) + ' tokens' : 'whole context (via strided / global hops)';
    WX.out(out, [['Pairs computed', `${c} of ${full}`], ['Saved vs full', WX.pct(1 - c / full, 0), 1], ['Reach after ' + v.L + ' layers', reach, 1], ['KV cache per layer', v.t === 'window' ? v.w + ' tokens (fixed)' : 'grows with context']]);
  };
  const v = WX.ctl(root, [{k: 't', l: 'Pattern', opts: [['full', 'Full causal'], ['window', 'Sliding window'], ['strided', 'Window + strided'], ['global', 'Window + global tokens']], v: 'window'}, {k: 'w', l: 'Window / stride', min: 2, max: 12, v: 4}, {k: 'L', l: 'Layers stacked', min: 1, max: 64, v: 32}], go);
  root.append(grid, out, WX.h('div', 'wnote', 'Rows are queries, columns are keys, for 24 tokens. Global tokens (orange) see and are seen by everyone.')); go(v);
};
WIDGETS.distill = root => {
  const C = [['2', 9.0], ['3', 5.6], ['7', 5.1], ['1', 3.0], ['8', 2.0], ['cat', -2.0]];
  const rows = WX.h('div'), out = WX.h('div', 'wout');
  const go = v => { const p = WX.softmax(C.map(c => c[1]), v.T), H = -p.reduce((a, x) => a + x * Math.log2(x), 0);
    WX.rows(rows, C.map((c, i) => ({l: 'class ' + c[0], v: p[i], max: 1, t: WX.pct(p[i], 2), hot: i === 0})));
    WX.out(out, [['Hard label says', '"2", nothing else'], ['Probability on 3 and 7', WX.pct(p[1] + p[2], 1), 1], ['Entropy', WX.f(H, 2) + ' bits'], ['Loss scale factor T²', WX.f(v.T * v.T, 1)]]); };
  const v = WX.ctl(root, [{k: 'T', l: 'Distillation temperature T', min: 0.5, max: 10, step: 0.1, v: 1}], go);
  root.append(rows, out, WX.h('div', 'wnote', 'Teacher logits for an image of a handwritten 2. At T = 1 the extra information is almost invisible; raise T to see that the teacher thinks 3 and 7 are plausible and "cat" is not.')); go(v);
};
WIDGETS.specdec = root => {
  const ch = WX.h('div'), out = WX.h('div', 'wout'), sim = WX.h('div', 'wtok'), btns = WX.h('div', 'wbtns', '<button type="button" class="wb" data-a="s">Simulate 10 rounds</button>');
  const E = (a, k) => a >= 1 ? k + 1 : (1 - a ** (k + 1)) / (1 - a);
  const go = v => {
    const toks = E(v.a, v.k), cost = 1 + v.k * v.c, sp = toks / cost;
    WX.out(out, [['Tokens per round (expected)', WX.f(toks, 2), 1], ['Cost per round (target-pass units)', WX.f(cost, 2)], ['Speedup', WX.f(sp, 2) + '×', 1]]);
    const series = [2, 4, 8].map((k, i) => ({l: 'k = ' + k, c: [WX.C1, WX.C2, WX.C3][i], pts: Array.from({length: 40}, (_, j) => { const a = 0.3 + j * 0.0175; return [a, E(a, k) / (1 + k * v.c)]; })}));
    WX.line(ch, series, {xl: 'Acceptance rate α', yl: 'Speedup', xmin: 0.3, xmax: 0.98, ymin: 0, fx: x => WX.f(x, 2), fy: y => WX.f(y, 1) + '×', marks: [{x: v.a}]});
  };
  btns.onclick = () => { let h = ''; for (let r = 0; r < 10; r++){ let acc = 0; while (acc < v.k && Math.random() < v.a) acc++; for (let i = 0; i < acc; i++) h += '<span style="background:var(--go-soft)">✓</span>'; h += `<span style="background:${acc < v.k ? 'var(--flame-soft)' : 'var(--accent-soft)'}">${acc < v.k ? '↺' : '+'}</span><span style="background:transparent">│</span>`; } sim.innerHTML = h; };
  const v = WX.ctl(root, [{k: 'a', l: 'Acceptance rate α', min: 0.3, max: 0.98, step: 0.01, v: 0.75}, {k: 'k', l: 'Draft length k', min: 1, max: 10, v: 4}, {k: 'c', l: 'Draft cost per token (× target)', min: 0.01, max: 0.3, step: 0.01, v: 0.05}], go);
  root.append(out, ch, btns, sim, WX.h('div', 'wnote', 'Expected tokens per round = (1 − α^(k+1)) / (1 − α), from Leviathan et al. ✓ accepted draft, ↺ corrected token from the target, + bonus token when all drafts pass.')); go(v);
};
WIDGETS.diagnose = root => {
  const S = [['ttft', 'First token is slow on long prompts'], ['tpot', 'Streaming is slow (time per token)'], ['thr', 'Cost per token too high / GPUs underused'], ['oom', 'Out of memory or tiny max batch'], ['long', 'Very long contexts are slow at every step'], ['qual', 'Need bigger-model quality at the same compute']];
  const F = {ttft: ['Prefix / prompt caching', 'Send less context (better retrieval)', 'FlashAttention + chunked prefill'], tpot: ['Weight quantization (int8 / 4-bit)', 'Speculative decoding', 'Smaller or distilled model', 'GQA / MLA model'], thr: ['Continuous batching', 'Paged attention', 'KV cache quantization'], oom: ['Quantize weights and KV cache', 'Paged attention', 'Tensor parallelism across GPUs'], long: ['Sliding-window / sparse attention model', 'Retrieval instead of stuffing context'], qual: ['Mixture-of-experts model', 'Distill from a larger teacher']};
  let on = new Set(['tpot']);
  const chips = WX.h('div', 'wchips'), res = WX.h('div');
  const draw = () => {
    chips.innerHTML = S.map(([k, t]) => `<button type="button" class="wchip${on.has(k) ? ' on' : ''}" data-k="${k}">${t}</button>`).join('');
    const cnt = {}; [...on].forEach(k => F[k].forEach(f => cnt[f] = (cnt[f] || 0) + 1));
    const r = Object.entries(cnt).sort((a, b) => b[1] - a[1]);
    res.innerHTML = r.length ? `<table class="wtbl"><tr><th>Try</th><th>Fixes</th></tr>${r.map(([f, c]) => `<tr class="${c > 1 ? 'hot' : ''}"><td>${f}</td><td class="n">${c} symptom${c > 1 ? 's' : ''}</td></tr>`).join('')}</table>` : '<div class="wnote">Pick at least one symptom.</div>';
    chips.querySelectorAll('[data-k]').forEach(b => b.onclick = () => { on.has(b.dataset.k) ? on.delete(b.dataset.k) : on.add(b.dataset.k); draw(); });
  };
  root.append(WX.h('div', 'wnote', 'Select every symptom you see. Fixes that address several symptoms rise to the top.'), chips, res); draw();
};
// ---------- module 8 ----------
WIDGETS.rewardhack = root => {
  const ch = WX.h('div'), out = WX.h('div', 'wout');
  const go = v => {
    // Gao et al. (2022) functional form for RL: gold(d) = d(a - b ln d), proxy(d) = d(a' - b' ln d) with a much smaller b', d = sqrt(KL)
    const a = 1, b = 0.51 - 0.002 * v.rm, ap = 0.35, bp = 0.012;
    const dmax = Math.sqrt(v.kl), gold = [], proxy = [];
    for (let i = 1; i <= 80; i++){ const d = i / 80 * 12; gold.push([d * d, Math.max(-2, d * (a - b * Math.log(d)))]); proxy.push([d * d, d * (ap - bp * Math.log(d))]); }
    const dstar = Math.exp(a / b - 1), peak = dstar * (a - b * Math.log(dstar)), now = dmax * (a - b * Math.log(dmax)), pnow = dmax * (ap - bp * Math.log(dmax));
    WX.line(ch, [{pts: proxy, c: WX.C2, l: 'Proxy reward (what the reward model says)', dash: 1}, {pts: gold, l: 'True quality (what humans would say)', dots: [[v.kl, now]]}], {xl: 'KL divergence from the starting model (nats)', yl: 'Reward', marks: [{x: Math.min(144, dstar * dstar), l: 'best stop'}, {x: v.kl, l: ''}], xmin: 0, xmax: 144, ymin: -2, fx: x => WX.f(x), fy: y => WX.f(y, 1)});
    WX.out(out, [['Proxy reward now', WX.f(pnow, 2)], ['True quality now', WX.f(now, 2), now < peak * 0.8], ['Best possible true quality', WX.f(peak, 2)], ['KL at the peak', WX.f(dstar * dstar, 0)]]);
  };
  const v = WX.ctl(root, [{k: 'kl', l: 'How hard RL optimizes (KL budget)', min: 1, max: 144, v: 30}, {k: 'rm', l: 'Reward model quality (data and size)', min: 0, max: 100, v: 40, fmt: x => x + '/100'}], go);
  root.append(ch, out, WX.h('div', 'wnote', 'The curve shape follows Gao, Schulman and Hilton (2022): proxy reward keeps rising while true quality peaks and then falls. Coefficients are illustrative. A better reward model pushes the peak to the right; it does not remove it. The KL penalty in RLHF is the brake that keeps training near the peak.')); go(v);
};
WIDGETS.selfconsistency = root => {
  const ch = WX.h('div'), out = WX.h('div', 'wout');
  const sim = (p, k, N, R, trials = 1500) => { let ok = 0; for (let t = 0; t < trials; t++){ const c = new Array(k + 1).fill(0); for (let i = 0; i < N; i++){ const r = R(); if (r < p) c[0]++; else c[1 + Math.floor((r - p) / (1 - p) * k * 0.999)]++; } const m = Math.max(...c); const w = c.map((x, i) => x === m ? i : -1).filter(i => i >= 0); if (w[Math.floor(R() * w.length)] === 0) ok++; } return ok / trials; };
  const go = v => {
    const R = WX.rng(7), p = v.p / 100, pts = [], Ns = [1, 3, 5, 7, 9, 15, 21, 31, 41];
    Ns.forEach(N => pts.push([N, sim(p, v.k, N, R)]));
    WX.line(ch, [{pts, l: 'Majority-vote accuracy', dots: pts.filter(q => q[0] === v.n)}, {pts: [[1, p], [41, p]], c: WX.CM, dash: 1, l: 'One sample'}], {xl: 'Samples per question (N)', yl: 'Accuracy', ymin: 0, ymax: 1, fx: x => WX.f(x), fy: y => WX.pct(y, 0)});
    const at = sim(p, v.k, v.n, R, 3000);
    WX.out(out, [['One sample', WX.pct(p)], [`Vote over ${v.n}`, WX.pct(at), 1], ['Cost multiplier', v.n + '×']]);
  };
  const v = WX.ctl(root, [{k: 'p', l: 'Chance one sample is right', min: 10, max: 90, v: 45, fmt: x => x + '%'}, {k: 'k', l: 'Distinct wrong answers the errors spread over', min: 1, max: 10, v: 4}, {k: 'n', l: 'N to highlight', min: 1, max: 41, step: 2, v: 9}], go);
  root.append(ch, out, WX.h('div', 'wnote', 'Monte Carlo simulation. Voting helps when the right answer is the most common single answer. Set p to 30% with one wrong answer and voting makes things worse: the errors agree with each other more than the right answers do.')); go(v);
};
WIDGETS.toolcall = root => {
  const S = {
    weather: {u: 'Do I need an umbrella in Pune tomorrow?', tool: 'get_forecast', args: {city: 'Pune', date: '2026-11-13'}, res: {rain_probability: 0.72, high_c: 27}, ans: 'Probably yes: the forecast gives a 72% chance of rain tomorrow in Pune.'},
    math: {u: 'What is 17.5% of 48,250, plus 1,200?', tool: 'calculator', args: {expression: '48250 * 0.175 + 1200'}, res: {value: 9643.75}, ans: '17.5% of 48,250 is 8,443.75; adding 1,200 gives 9,643.75.'},
    db: {u: 'How many orders did customer 4411 place this year?', tool: 'run_sql', args: {query: "SELECT COUNT(*) FROM orders WHERE customer_id = 4411 AND created_at >= '2026-01-01'"}, res: {rows: [[37]]}, ans: 'Customer 4411 has placed 37 orders since 1 January 2026.'},
  };
  let k = 'weather', step = 0;
  const box = WX.h('div', 'wlog'), btns = WX.h('div', 'wbtns');
  const steps = s => [
    ['App → model', `user: ${s.u}\ntools: [${Object.keys(S).map(x => S[x].tool).join(', ')}] (name, description, JSON schema each)`],
    ['Model → app', `stop_reason: tool_use\n{"name": "${s.tool}", "input": ${JSON.stringify(s.args)}}`],
    ['App runs the tool', `your code validates the arguments, checks permissions, calls ${s.tool}(...)\nresult: ${JSON.stringify(s.res)}`],
    ['App → model', `tool_result: ${JSON.stringify(s.res)}\n(appended to the conversation, then the model is called again)`],
    ['Model → user', s.ans],
  ];
  const draw = () => { const st = steps(S[k]); box.textContent = st.slice(0, step + 1).map(([a, b], i) => `── ${i + 1}. ${a}\n${b}`).join('\n\n'); btns.querySelector('[data-a=n]').disabled = step >= st.length - 1; };
  btns.innerHTML = Object.keys(S).map(x => `<button class="wb${x === k ? ' on' : ''}" data-k="${x}">${x}</button>`).join('') + '<button class="wb" data-a="n">Next step ›</button><button class="wb" data-a="r">Restart</button>';
  btns.onclick = e => { const b = e.target.closest('button'); if (!b) return; if (b.dataset.k){ k = b.dataset.k; step = 0; btns.querySelectorAll('[data-k]').forEach(x => x.classList.toggle('on', x === b)); } if (b.dataset.a === 'n') step++; if (b.dataset.a === 'r') step = 0; draw(); };
  root.append(btns, box, WX.h('div', 'wnote', 'The model never runs anything. It only writes a structured request; your code decides whether to execute it. The message shapes are simplified from the Anthropic and OpenAI tool-use APIs.')); draw();
};
WIDGETS.tot = root => {
  const ops = [['+', (a, b) => a + b], ['−', (a, b) => a - b], ['×', (a, b) => a * b], ['÷', (a, b) => b ? a / b : NaN]];
  const memo = new Map();
  const kids = st => { const r = []; for (let i = 0; i < st.n.length; i++) for (let j = 0; j < st.n.length; j++){ if (i === j) continue; ops.forEach(([o, f]) => { if ((o === '+' || o === '×') && j < i) return; const x = f(st.n[i], st.n[j]); if (!isFinite(x)) return; const n = [], e = []; st.n.forEach((v, q) => { if (q !== i && q !== j){ n.push(v); e.push(st.e[q]); } }); n.push(x); e.push(`(${st.e[i]} ${o} ${st.e[j]})`); r.push({n, e}); }); } return r; };
  const key = n => n.map(x => Math.round(x * 1e6) / 1e6).sort((a, b) => a - b).join(',');
  const can = n => { if (n.length === 1) return Math.abs(n[0] - 24) < 1e-6; const k = key(n); if (memo.has(k)) return memo.get(k); const r = kids({n, e: n.map(String)}).some(c => can(c.n)); memo.set(k, r); return r; };
  const fmt = x => Number.isInteger(Math.round(x * 1e6) / 1e6) ? String(Math.round(x)) : WX.f(x, 2);
  const lv = WX.h('div'), out = WX.h('div', 'wout'), sol = WX.h('div', 'wlog');
  const go = v => {
    const R = WX.rng(v.seed * 31 + 5), err = v.err / 100;
    let front = [{n: [v.a, v.b, v.c, v.d], e: [v.a, v.b, v.c, v.d].map(String)}], html = '', gen = 0, calls = 0, found = null;
    for (let L = 1; L <= 3; L++){
      const seen = new Set(), all = [];
      front.forEach(s => kids(s).forEach(c => { const k = key(c.n); if (!seen.has(k)){ seen.add(k); all.push(c); } }));
      gen += all.length; calls += all.length;
      all.forEach(c => { const t = can(c.n); c.truth = t; c.say = R() < err ? !t : t; c.r = R(); });
      all.sort((x, y) => (y.say - x.say) || (x.r - y.r));
      front = all.slice(0, v.bw);
      html += `<div style="margin-bottom:10px"><div class="wnote">Level ${L}: ${all.length} candidate states generated and rated, ${all.filter(c => c.say).length} rated "possible", keep ${front.length}</div><div class="wchips">${front.map(c => `<span class="wchip${c.truth ? ' on' : ''}" title="${c.truth ? 'really solvable' : 'really a dead end'}">${c.n.map(fmt).join('  ')} ${c.say ? '✓' : '✗'}${c.say !== c.truth ? ' (misjudged)' : ''}</span>`).join('')}</div></div>`;
      if (L === 3) found = front.find(c => Math.abs(c.n[0] - 24) < 1e-6);
    }
    lv.innerHTML = html;
    sol.textContent = found ? `Solved: ${found.e[0].replace(/^\((.*)\)$/, '$1')} = 24` : (can([v.a, v.b, v.c, v.d]) ? 'Search failed: every solvable state was pruned. Raise the breadth or lower the evaluator error.' : 'No solution exists for these four numbers, so every path is a dead end.');
    WX.out(out, [['States generated', gen], ['Model calls (approx.)', calls + 3], ['Solvable?', can([v.a, v.b, v.c, v.d]) ? 'yes' : 'no'], ['Found', found ? 'yes' : 'no', !found]]);
  };
  const N = (k, l, val) => ({k, l, min: 1, max: 13, v: val});
  const v = WX.ctl(root, [N('a', 'Number 1', 4), N('b', 'Number 2', 9), N('c', 'Number 3', 10), N('d', 'Number 4', 13), {k: 'bw', l: 'Breadth b (states kept per level)', min: 1, max: 10, v: 5}, {k: 'err', l: 'Evaluator error rate', min: 0, max: 50, v: 15, fmt: x => x + '%'}, {k: 'seed', l: 'Random seed', min: 1, max: 20, v: 1}], go);
  root.append(lv, sol, out, WX.h('div', 'wnote', 'A real breadth-first search over Game of 24. The evaluator here is an exact solver whose answer is flipped at the chosen error rate, standing in for an LLM that rates states "sure / impossible". Highlighted chips are truly solvable. Try b = 1 with 20% error: that is close to plain chain of thought.')); go(v);
};
WIDGETS.verify = root => {
  const ch = WX.h('div'), out = WX.h('div', 'wout');
  const C = (n, k) => { let r = 1; for (let i = 1; i <= k; i++) r = r * (n - k + i) / i; return r; };
  const P = (n, p, tpr, fpr) => { let s = 0; for (let k = 0; k <= n; k++){ const pk = C(n, k) * p ** k * (1 - p) ** (n - k); // k correct of n
      let q = 0; for (let a = 0; a <= k; a++) for (let b = 0; b <= n - k; b++){ const pa = C(k, a) * tpr ** a * (1 - tpr) ** (k - a) * C(n - k, b) * fpr ** b * (1 - fpr) ** (n - k - b); q += pa * (a + b ? a / (a + b) : k / n); }
      s += pk * q; } return s; };
  const go = v => {
    const p = v.p / 100, t = v.t / 100, f = v.f / 100, pts = [], ideal = [];
    for (let n = 1; n <= 16; n++){ pts.push([n, P(n, p, t, f)]); ideal.push([n, 1 - (1 - p) ** n]); }
    WX.line(ch, [{pts: ideal, c: WX.CM, dash: 1, l: 'Perfect verifier (pass@n)'}, {pts, l: 'Your verifier', dots: [[v.n, P(v.n, p, t, f)]]}], {xl: 'Candidates generated (n)', yl: 'Chance the chosen answer is right', ymin: 0, ymax: 1, fx: x => WX.f(x), fy: y => WX.pct(y, 0)});
    WX.out(out, [['Single attempt', WX.pct(p)], [`Best of ${v.n}`, WX.pct(P(v.n, p, t, f)), 1], ['Ceiling (pass@' + v.n + ')', WX.pct(1 - (1 - p) ** v.n)]]);
  };
  const v = WX.ctl(root, [{k: 'p', l: 'Generator accuracy per attempt', min: 5, max: 90, v: 30, fmt: x => x + '%'}, {k: 't', l: 'Verifier accepts a right answer', min: 50, max: 100, v: 90, fmt: x => x + '%'}, {k: 'f', l: 'Verifier accepts a wrong answer', min: 0, max: 60, v: 15, fmt: x => x + '%'}, {k: 'n', l: 'n to highlight', min: 1, max: 16, v: 8}], go);
  root.append(ch, out, WX.h('div', 'wnote', 'Exact calculation: generate n answers, keep the ones the verifier accepts, pick one at random (or any answer if none pass). Raise the false-accept rate and the gain from more samples flattens fast. The verifier, not the generator, sets the ceiling.')); go(v);
};
// ---------- module 9 ----------
WIDGETS.context = root => {
  const bar = WX.h('div'), out = WX.h('div', 'wout'), note = WX.h('div', 'wnote');
  const go = v => {
    const parts = [['System prompt', v.sys, WX.C1], ['Tool definitions', v.tools * 350, WX.C4], ['Conversation history', v.turns * 450, WX.C2], ['Retrieved documents', v.docs * 800, WX.C3], ['Reserved for the answer', v.outp, WX.CM]];
    const used = parts.reduce((a, p) => a + p[1], 0), win = v.win * 1000, over = used > win;
    bar.innerHTML = `<div style="display:flex;height:28px;border-radius:7px;overflow:hidden;border:1px solid var(--line2)">${parts.map(([l, x, c]) => `<i title="${l}: ${WX.f(x)} tokens" style="display:block;width:${x / Math.max(win, used) * 100}%;background:${c}"></i>`).join('')}</div><div class="wnote" style="margin-top:6px">${parts.map(([l, x, c]) => `<span style="display:inline-flex;align-items:center;gap:5px;margin-right:12px"><i style="width:10px;height:10px;border-radius:2px;background:${c};display:inline-block"></i>${l} ${WX.big(x)}</span>`).join('')}</div>`;
    const inTok = used - v.outp;
    WX.out(out, [['Tokens used', `${WX.big(used)} / ${WX.big(win)}`, over], ['Input cost per call', '$' + WX.f(inTok / 1e6 * 3, 4)], ['Cost for a 20-step agent run', '$' + WX.f(inTok / 1e6 * 3 * 20, 2)], ['Share that is tool schemas', WX.pct(v.tools * 350 / used)]]);
    note.textContent = over ? 'Over the window: the request fails or the oldest content is cut. Summarize history, retrieve fewer chunks, or load tools on demand.' : used > win * 0.7 ? 'Fits, but long contexts dilute attention: facts in the middle of a long prompt are recalled less reliably. Trim before you must.' : 'Comfortable. Every token here is resent on every agent step, so the per-run cost grows with steps × context.';
  };
  const v = WX.ctl(root, [{k: 'win', l: 'Context window (k tokens)', min: 8, max: 1000, step: 8, v: 200, fmt: x => x + 'k'}, {k: 'sys', l: 'System prompt tokens', min: 200, max: 20000, step: 200, v: 2500}, {k: 'tools', l: 'Tools loaded (~350 tokens each)', min: 0, max: 120, v: 25}, {k: 'turns', l: 'History turns kept (~450 each)', min: 0, max: 300, v: 40}, {k: 'docs', l: 'Retrieved chunks (~800 each)', min: 0, max: 150, v: 12}, {k: 'outp', l: 'Reserved for output', min: 500, max: 64000, step: 500, v: 8000}], go);
  root.append(bar, out, note, WX.h('div', 'wnote', 'Input priced at an illustrative $3 per million tokens, without prompt caching.')); go(v);
};
WIDGETS.reliability = root => {
  const ch = WX.h('div'), out = WX.h('div', 'wout');
  const go = v => {
    const p = v.p / 100, step = v.r ? 1 - (1 - p) * ((1 - v.d / 100) + (v.d / 100) * (1 - p) ** v.r) : p;
    const a = [], b = []; for (let n = 1; n <= 50; n++){ a.push([n, p ** n]); b.push([n, step ** n]); }
    WX.line(ch, [{pts: a, c: WX.CM, dash: 1, l: 'No error handling'}, {pts: b, l: `Retry up to ${v.r}× when the error is detected`, dots: [[v.n, step ** v.n]]}], {xl: 'Steps in the task', yl: 'Task success', ymin: 0, ymax: 1, fx: x => WX.f(x), fy: y => WX.pct(y, 0), marks: [{x: v.n}]});
    WX.out(out, [['Per-step success', WX.pct(p)], ['Effective per step with retries', WX.pct(step)], [`Task success at ${v.n} steps`, WX.pct(step ** v.n), step ** v.n < .5], ['Without retries', WX.pct(p ** v.n)]]);
  };
  const v = WX.ctl(root, [{k: 'p', l: 'Per-step success', min: 80, max: 99.9, step: .1, v: 95, fmt: x => WX.f(x, 1) + '%'}, {k: 'd', l: 'Share of errors the loop detects', min: 0, max: 100, v: 60, fmt: x => x + '%'}, {k: 'r', l: 'Retries per step', min: 0, max: 5, v: 2}, {k: 'n', l: 'Steps to highlight', min: 1, max: 50, v: 20}], go);
  root.append(ch, out, WX.h('div', 'wnote', 'Assumes independent steps, which is optimistic: real errors cluster. The lesson holds anyway: long chains need checks after each step, because undetected errors compound and cannot be retried.')); go(v);
};
WIDGETS.mcpcount = root => {
  const rows = WX.h('div'), out = WX.h('div', 'wout');
  const go = v => {
    const m = v.a * v.t, n = v.a + v.t;
    WX.rows(rows, [{l: 'Custom glue', v: m, t: WX.f(m), c: WX.C2}, {l: 'With MCP', v: n, t: WX.f(n)}], v => WX.f(v));
    WX.out(out, [['Integrations to build (N × M)', WX.f(m), 1], ['With a shared protocol (N + M)', WX.f(n)], ['Work saved', WX.pct(1 - n / m, 0)]]);
  };
  const v = WX.ctl(root, [{k: 'a', l: 'AI apps (clients)', min: 1, max: 30, v: 6}, {k: 't', l: 'Tools and data sources (servers)', min: 1, max: 60, v: 15}], go);
  root.append(rows, out, WX.h('div', 'wnote', 'The same argument that made USB and the Language Server Protocol win: each side implements the protocol once. It does not remove the work of writing a good server.')); go(v);
};
WIDGETS.evalcalc = root => {
  const ch = WX.h('div'), out = WX.h('div', 'wout');
  const go = v => {
    const p = v.p / 100, z = 1.96, moe = n => z * Math.sqrt(p * (1 - p) / n);
    const pts = []; for (let n = 10; n <= 2000; n = Math.round(n * 1.25)) pts.push([n, moe(n)]);
    const need = d => Math.ceil(2 * (z + 0.84) ** 2 * p * (1 - p) / (d * d));
    WX.line(ch, [{pts, l: '± margin of error (95%)', dots: [[v.n, moe(v.n)]]}], {xl: 'Eval cases', yl: '± points', xlog: 1, ymin: 0, fx: x => WX.f(x), fy: y => WX.pct(y, 0)});
    WX.out(out, [['Measured pass rate', WX.pct(p, 0)], [`Margin with ${v.n} cases`, '± ' + WX.pct(moe(v.n)), moe(v.n) > v.d / 100], [`Cases per variant to detect a ${v.d}-point change`, WX.f(need(v.d / 100))]]);
  };
  const v = WX.ctl(root, [{k: 'p', l: 'Pass rate', min: 5, max: 95, v: 80, fmt: x => x + '%'}, {k: 'n', l: 'Eval cases', min: 10, max: 2000, step: 10, v: 100}, {k: 'd', l: 'Change you want to detect (points)', min: 1, max: 20, v: 5}], go);
  root.append(ch, out, WX.h('div', 'wnote', 'Normal approximation, 95% confidence, 80% power, independent cases. With 100 cases at 80% pass rate the uncertainty is about ±8 points, so a 3-point "improvement" is noise.')); go(v);
};

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
