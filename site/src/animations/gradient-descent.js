// Gradient descent on loss(w) = w². Each step moves w by -lr * 2w; the learning-rate slider shows
// slow convergence (small lr), the one-step jump (lr = 0.5), oscillation (0.5 < lr < 1) and divergence (lr > 1).
const W = 640, H = 340, X0 = 60, X1 = 600, Y0 = 290, Y1 = 30, WMAX = 3.2, LMAX = WMAX * WMAX;
const px = w => X0 + (w + WMAX) / (2 * WMAX) * (X1 - X0);
const py = L => Y0 - Math.min(L, LMAX * 1.15) / LMAX * (Y0 - Y1);
const STEPS = 8, W0 = 2.6;
const path = lr => { const ws = [W0]; for (let i = 0; i < STEPS; i++) ws.push(ws[i] - lr * 2 * ws[i]); return ws; };
const fmt = x => Math.abs(x) >= 100 ? x.toExponential(1) : x.toFixed(2);

export default {
  title: 'Gradient descent on a parabola',
  w: W, h: H, steps: STEPS,
  controls: [{id: 'lr', label: 'Learning rate', min: 0.05, max: 1.1, step: 0.05, v: 0.3, f: v => v.toFixed(2)}],
  frame(s, t, {lr}) {
    const ws = path(lr), e = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;   // ease in-out
    const w = ws[s] + (ws[s + 1] - ws[s]) * e, curve = [];
    for (let i = 0; i <= 120; i++) { const x = -WMAX + i / 60 * WMAX; curve.push(`${i ? 'L' : 'M'}${px(x).toFixed(1)} ${py(x * x).toFixed(1)}`); }
    const trail = ws.slice(0, s + 1).map((x, i) => `<circle cx="${px(x)}" cy="${py(x * x)}" r="4" fill="var(--accent)" opacity="${0.25 + 0.5 * i / STEPS}"/>`).join('');
    const out = Math.abs(w) > WMAX;
    return `<line x1="${X0}" y1="${Y0}" x2="${X1}" y2="${Y0}" stroke="var(--line2)" stroke-width="1.5"/>
      <line x1="${px(0)}" y1="${Y0}" x2="${px(0)}" y2="${Y1 - 10}" stroke="var(--line2)" stroke-width="1.5"/>
      <path d="${curve.join('')}" fill="none" stroke="var(--accent)" stroke-width="3"/>
      <text x="${X1}" y="${Y0 + 22}" text-anchor="end" font-size="13" fill="var(--muted)">w</text>
      <text x="${px(0) + 8}" y="${Y1}" font-size="13" fill="var(--muted)">loss(w) = w²</text>
      ${trail}
      ${out ? `<text x="${W / 2}" y="${Y1 + 30}" text-anchor="middle" font-size="15" font-weight="600" fill="var(--bad)">diverging: |w| keeps growing</text>`
            : `<circle cx="${px(w)}" cy="${py(w * w)}" r="8" fill="var(--flame)" stroke="var(--card)" stroke-width="2"/>`}`;
  },
  caption: (s, {lr}) => { const ws = path(lr), w = ws[s]; return `Step ${s}/${STEPS}: w = ${fmt(w)}, loss = ${fmt(w * w)}, gradient = ${fmt(2 * w)}`; },
};
