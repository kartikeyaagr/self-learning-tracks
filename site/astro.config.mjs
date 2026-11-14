import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkDiagrams from './src/plugins/remark-diagrams.mjs';
import rehypeTables from './src/plugins/rehype-tables.mjs';

// Everything renders at build time: Markdown (remark + GFM), maths (KaTeX), code (Shiki), diagrams (house SVG).
// Astro 7 defaults to its native Sätteri pipeline; it has no maths support yet (0.x), so we keep unified().
export default defineConfig({
  trailingSlash: 'always',
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath, remarkDiagrams],
      rehypePlugins: [[rehypeKatex, {throwOnError: false}], rehypeTables],
      smartypants: false,                       // lessons already use typographic punctuation; keep code-ish text intact
    }),
    shikiConfig: {themes: {light: 'github-light', dark: 'github-dark'}},
  },
});
