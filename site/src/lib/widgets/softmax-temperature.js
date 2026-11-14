// Lesson widget "softmax-temperature": attention weights of the query "it" over one sentence, as bars of
// softmax(score / T) from fixed scores, with a temperature slider (0.25–3). Styles: styles/screens/lesson.css (.sw).
// Mounted by the lesson player on <div data-widget="softmax-temperature">; mount() returns nothing, state is local.
export const WORDS = ['The', 'animal', 'didn’t', 'cross', 'the', 'street', 'because', 'it', 'was', 'too', 'tired'];
export const SCORES = [0.2, 3.0, 0.3, 0.6, 0.1, 2.2, 0.4, 1.0, 0.5, 0.3, 1.6];   // hand-set: "it" refers to "animal"
export const QUERY = 7;

// Stable softmax with temperature: subtract the max first (only differences matter).
export function softmax(s, T = 1) {
  const m = Math.max(...s), e = s.map(x => Math.exp((x - m) / T)), z = e.reduce((a, b) => a + b, 0);
  return e.map(x => x / z);
}

export function mount(el) {
  el.innerHTML = `<section class="sw" aria-label="Attention weights of “it”">
    ${WORDS.map((w, i) => `<div class="sw-row" style="--i:${i}"><span class="w${i === QUERY ? ' q' : ''}">${w}</span>
      <span class="tr"><span class="b"></span></span><span class="mono p"></span></div>`).join('')}
    <label class="sw-t"><span>Temperature</span><input type="range" min="0.25" max="3" step="0.25" value="1" aria-valuetext="1.00">
      <span class="mono v">1.00</span></label></section>`;
  const rows = [...el.querySelectorAll('.sw-row')], input = el.querySelector('input'), out = el.querySelector('.v');
  const draw = () => {
    const T = +input.value, p = softmax(SCORES, T), top = p.indexOf(Math.max(...p));
    rows.forEach((r, i) => { const pct = Math.round(p[i] * 100);
      r.querySelector('.b').style.width = pct + '%'; r.querySelector('.p').textContent = pct + '%'; r.classList.toggle('top', i === top); });
    out.textContent = T.toFixed(2); input.setAttribute('aria-valuetext', T.toFixed(2));
  };
  input.addEventListener('input', draw);
  draw();
}
