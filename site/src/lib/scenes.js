// SVG scene player. A scene (src/animations/<id>.js) exports:
//   { title, w, h, steps, controls?: [{id, label, min, max, step, v, f?}], frame(step, t, params) → SVG inner markup,
//     caption?(step, params) → text for the status line }
// step is 0..steps-1 and t is 0..1 within that step. Colours come from CSS variables, so light/dark just work.
// Embedded in lessons with a ```dg-anim <id> | caption``` block. Reduced motion shows the final frame, no autoplay.
const SCENES = import.meta.glob('../animations/*.js', {eager: true});
const RM = matchMedia('(prefers-reduced-motion: reduce)').matches;
const DUR = 900;                                   // ms per step

export function initScenes(root) {
  root.querySelectorAll('figure.anim').forEach(fig => {
    const mod = SCENES[`../animations/${fig.dataset.anim}.js`]?.default, stage = fig.querySelector('.anim-stage');
    if (!mod) { stage.textContent = 'Animation unavailable.'; return; }
    const P = Object.fromEntries((mod.controls || []).map(c => [c.id, c.v]));
    let pos = RM ? mod.steps : 0, playing = false, last = 0;  // pos = steps done, fractional
    stage.innerHTML = `<svg viewBox="0 0 ${mod.w} ${mod.h}" role="img" aria-label="${mod.title}" style="color:var(--ink);font-family:var(--body)"></svg>
      ${(mod.controls || []).map(c => `<label class="wg-s"><span>${c.label} <output></output></span><input type="range" data-id="${c.id}" min="${c.min}" max="${c.max}" step="${c.step}" value="${c.v}"></label>`).join('')}
      <div class="anim-ctl"><button type="button" class="btn ghost" data-a="play" style="min-height:36px">Play</button>
        <button type="button" class="btn ghost" data-a="prev" aria-label="Previous step" style="min-height:36px">‹</button>
        <input type="range" min="0" max="${mod.steps}" step="0.01" value="${pos}" aria-label="Scrub">
        <button type="button" class="btn ghost" data-a="next" aria-label="Next step" style="min-height:36px">›</button></div>
      <div class="dg-st" aria-live="polite" style="text-align:center;margin-top:6px"></div>`;
    const svg = stage.querySelector('svg'), scrub = stage.querySelector('.anim-ctl input'), st = stage.querySelector('.dg-st'), playB = stage.querySelector('[data-a="play"]');
    const draw = () => {
      const s = Math.max(0, Math.min(Math.floor(pos), mod.steps - 1)), t = pos >= mod.steps ? 1 : pos - Math.floor(pos);
      svg.innerHTML = mod.frame(s, t, P); scrub.value = pos;
      if (mod.caption) st.textContent = mod.caption(Math.min(Math.round(pos), mod.steps), P);
    };
    // rAF timestamps can predate performance.now(), so clamp the first delta at 0 (a negative step would draw step -1).
    const loop = ts => { if (!playing) return; pos = Math.min(mod.steps, pos + Math.max(0, ts - last) / DUR); last = ts; draw();
      if (pos >= mod.steps) { playing = false; playB.textContent = 'Replay'; } else requestAnimationFrame(loop); };
    const play = () => { if (pos >= mod.steps) pos = 0; playing = true; playB.textContent = 'Pause'; last = performance.now(); requestAnimationFrame(loop); };
    const stop = () => { playing = false; playB.textContent = 'Play'; };
    playB.onclick = () => playing ? stop() : play();
    stage.querySelector('[data-a="prev"]').onclick = () => { stop(); pos = Math.max(0, Math.ceil(pos) - 1); draw(); };
    stage.querySelector('[data-a="next"]').onclick = () => { stop(); pos = Math.min(mod.steps, Math.floor(pos) + 1); draw(); };
    scrub.oninput = () => { stop(); pos = +scrub.value; draw(); };
    stage.querySelectorAll('.wg-s input').forEach(i => { const c = mod.controls.find(x => x.id === i.dataset.id), o = i.previousElementSibling.querySelector('output');
      const set = () => { P[c.id] = +i.value; o.textContent = c.f ? c.f(+i.value) : i.value; draw(); }; i.oninput = set; set(); });
    draw();
    // autoplay once when scrolled into view
    if (!RM) new IntersectionObserver((es, ob) => { if (es[0].isIntersecting) { ob.disconnect(); play(); } }, {threshold: .6}).observe(fig);
  });
}
