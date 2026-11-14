// Render Markdown strings (problem statements, hints, explanations, quiz text) with the same pipeline as lessons.
import { unified } from '@astrojs/markdown-remark';
import { remarkPlugins, rehypePlugins, shikiConfig } from './markdown-config.mjs';
let renderer;
export async function md(s) {
  renderer ??= await unified({remarkPlugins, rehypePlugins, smartypants: false}).createRenderer({shikiConfig, syntaxHighlight: 'shiki'});
  return (await renderer.render(String(s || ''))).code;
}
