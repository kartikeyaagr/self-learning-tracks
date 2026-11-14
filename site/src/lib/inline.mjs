// Inline markdown for build-time rendering: **bold**, `code`, *em*, and $maths$ rendered with KaTeX.
import katex from 'katex';
import { esc, inlBasic } from './inline-basic.mjs';
export { esc };
export const inl = s => inlBasic(s, m => katex.renderToString(m, {throwOnError: false}));
