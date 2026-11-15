// Phones hide the desktop header (and its menu button), so pages with a sidebar get a "Contents" button at the top of
// the content that opens the same drawer (#side, closed by Site.astro's #scrim handler). Shown only ≤640px (legacy.css).
const side = document.getElementById('side'), view = document.getElementById('view');
if (side && view && !document.getElementById('contents')) {
  const b = document.createElement('button');
  b.type = 'button'; b.id = 'contents'; b.className = 'u-contents'; b.textContent = 'Contents';
  b.addEventListener('click', () => { side.classList.add('open'); document.getElementById('scrim')?.classList.add('on'); });
  side.addEventListener('click', e => { if (e.target.closest('a')) { side.classList.remove('open'); document.getElementById('scrim')?.classList.remove('on'); } });
  view.prepend(b);
}
