// The Markdown pipeline, shared by astro.config.mjs (lessons) and lib/md.mjs (practice text).
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkDiagrams from '../plugins/remark-diagrams.mjs';
import rehypeTables from '../plugins/rehype-tables.mjs';
export const remarkPlugins = [remarkMath, remarkDiagrams];
export const rehypePlugins = [[rehypeKatex, {throwOnError: false}], rehypeTables];
export const shikiConfig = {themes: {light: 'github-light', dark: 'github-dark'}};
