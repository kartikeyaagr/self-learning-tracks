// Shared by every course page (injected by build_hub.py), so the site reads as one product:
//  1. site-wide nav in the course header: Roadmap · Practice · Study plans
//  2. exported courses (whose engines predate the Practice Lab) get "Practice this" on lesson pages and
//     study-plan links on module pages, from practice/links.json and SITE_PLANS below.
// Courses built from engine/ render both natively and set SITE.native = true.
(() => {
  const S = window.SITE || {};
  const css = `.sitenav{display:flex;gap:2px;align-items:center;margin-left:6px}
.sitenav a{font:500 13px var(--body,system-ui);color:var(--muted);text-decoration:none;padding:6px 9px;border-radius:8px;white-space:nowrap}
.sitenav a:hover{color:var(--ink);background:var(--panel)}
@media (max-width:1100px){.sitenav{display:none}}
.site-card{margin-top:44px;padding:26px;background:var(--card);border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow)}
.site-card h3{margin:0 0 6px;font-size:21px;letter-spacing:-.03em}
.site-links{display:flex;flex-direction:column;gap:8px;margin-top:12px}
.site-links a{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:12px 14px;border-radius:12px;border:1px solid var(--line);background:var(--bg);text-decoration:none;font-size:15px;color:var(--ink)}
.site-links a:hover{border-color:var(--accent)}
.site-links .d{font:500 11.5px var(--mono,monospace);color:var(--muted);white-space:nowrap}
.site-links .d.Easy{color:var(--go)} .site-links .d.Medium{color:var(--gold)} .site-links .d.Hard{color:var(--bad)}`;
  const st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);

  // 1. nav after the brand (the hub button is injected before it)
  const brand = document.querySelector('header.top .brand');
  if (brand && !document.querySelector('.sitenav')) {
    const up = S.root || '../';
    const nav = document.createElement('nav'); nav.className = 'sitenav'; nav.setAttribute('aria-label', 'Site');
    nav.innerHTML = `<a href="${up}index.html">Roadmap</a><a href="${up}practice/index.html">Practice</a><a href="${up}practice/index.html#plans">Study plans</a>`;
    brand.after(nav);
  }
  if (S.native || !S.key) return;

  // 2. practice + plan links for exported courses
  const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
  let links = null;
  fetch('../practice/links.json').then(r => r.ok ? r.json() : {}).then(j => { links = j; draw(); }).catch(() => { links = {}; });
  const card = (lab, title, rows) => `<section class="site-card" data-site><div class="lab">${lab}</div><h3>${title}</h3><div class="site-links">${rows}</div></section>`;
  function draw() {
    if (!links) return;
    const h = location.hash.slice(1), view = document.getElementById('view');
    if (!view || view.querySelector('[data-site]')) return;
    let m = h.match(/^l([0-9A-Z]{2})-(\d\d)$/);
    if (m) {
      const ps = links[`${S.key}:${m[1]}.${m[2]}`] || [];
      const anchor = view.querySelector('.done-zone') || view.querySelector('article');
      if (ps.length && anchor) anchor.insertAdjacentHTML(anchor.classList.contains('done-zone') ? 'beforebegin' : 'beforeend',
        card('Practice this', 'Problems that use this lesson', ps.map(([slug, t, d]) => `<a href="../practice/index.html#l${slug}"><span>${esc(t)}</span><span class="d ${d}">${d}</span></a>`).join('')));
      return;
    }
    m = h.match(/^m([0-9A-Z]{2})$/);
    const plans = m && (S.plans || {})[m[1]];
    const nav = view.querySelector('.mnav');
    if (plans && plans.length && nav) nav.insertAdjacentHTML('beforebegin', card('Practise this module', 'Study plan sections',
      plans.map(([id, t, sec]) => `<a href="../practice/index.html#plan-${id}"><span>${esc(t)}${sec ? ' · ' + esc(sec) : ''}</span><span class="d">Plan →</span></a>`).join('')).replace('class="site-card"', 'class="site-card" style="max-width:760px;margin:30px auto 0"'));
  }
  // Pages re-render #view on navigation; redraw after each render.
  const view = document.getElementById('view');
  if (view) new MutationObserver(() => draw()).observe(view, {childList: true});
  addEventListener('hashchange', () => setTimeout(draw, 0));
})();
