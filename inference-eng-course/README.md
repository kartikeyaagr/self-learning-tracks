# Inference Engineering Course (static export)

A single-page site with no build step and no server code. The files are:

- `index.html`: the whole course, including all lesson data, styles and scripts.
- `favicon.svg`: the tab icon.

## Progress storage

Progress, notes, grades, streaks and the review deck are saved in the browser's `localStorage` under the key `ieng`. Each visitor's progress stays on their own device and browser. Clearing site data resets it.

The claude.ai version syncs progress through the artifact database. That sync is turned off here automatically.

## Deploy

Upload both files to any static host. Pick one:

- **Netlify:** drag the folder onto https://app.netlify.com/drop
- **Cloudflare Pages:** run `npx wrangler pages deploy . --project-name inference-course`
- **GitHub Pages:** push both files to a repo, then go to Settings > Pages > Deploy from branch (root).
- **Vercel:** run `npx vercel --prod` from this folder.
- **Your own server:** serve the folder as static files, for example `python3 -m http.server 8080`.

Routing uses the URL hash (`#m03`, `#l03-02`, `#review`), so no rewrite rules are needed.

## External requests

- The fonts (Geist, JetBrains Mono, Instrument Serif) load from Google Fonts. Offline, the page falls back to system fonts.
- Each lesson's source links point to external sites.
