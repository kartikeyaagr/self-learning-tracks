import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import { remarkPlugins, rehypePlugins, shikiConfig } from './src/lib/markdown-config.mjs';
import { lessonHref } from './src/plugins/remark-lesson-links.mjs';
import retired from './redirects.json' with { type: 'json' };

// Retired lessons and modules keep working: old URL → its new home (lesson: refs resolve like lesson links).
const { _comment, ...moved } = retired;
const base = '/self-learning-tracks';   // Astro does not prefix redirect targets with base, so targets carry it
const redirects = Object.fromEntries(Object.entries(moved).map(([from, to]) =>
  [from, to.startsWith('lesson:') ? lessonHref(to.slice(7), 'in redirects.json') : base + to]));

// Everything renders at build time: Markdown (remark + GFM), maths (KaTeX), code (Shiki), diagrams (house SVG).
// Astro 7 defaults to its native Sätteri pipeline; it has no maths support yet (0.x), so we keep unified().
export default defineConfig({
  site: 'https://kartikeyaagr.github.io',
  base,      // GitHub Pages project site; every internal link goes through lib/url.mjs u()
  trailingSlash: 'always',
  redirects,
  markdown: {
    processor: unified({remarkPlugins, rehypePlugins, smartypants: false}),   // lessons already use typographic punctuation
    shikiConfig,
  },
});
