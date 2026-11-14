// Every internal link goes through u(): the site is served under a base path on GitHub Pages (/self-learning-tracks/).
const BASE = import.meta.env.BASE_URL.replace(/\/$/, '');
export const u = p => BASE + p;
