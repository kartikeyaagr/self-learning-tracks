import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import { remarkPlugins, rehypePlugins, shikiConfig } from './src/lib/markdown-config.mjs';

// Everything renders at build time: Markdown (remark + GFM), maths (KaTeX), code (Shiki), diagrams (house SVG).
// Astro 7 defaults to its native Sätteri pipeline; it has no maths support yet (0.x), so we keep unified().
export default defineConfig({
  site: 'https://kartikeyaagr.github.io',
  base: '/self-learning-tracks',      // GitHub Pages project site; every internal link goes through lib/url.mjs u()
  trailingSlash: 'always',
  markdown: {
    processor: unified({remarkPlugins, rehypePlugins, smartypants: false}),   // lessons already use typographic punctuation
    shikiConfig,
  },
});
