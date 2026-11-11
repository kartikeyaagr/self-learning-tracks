# AI Engineering Course (static site)

A self-contained copy of the AI Engineering course. No build step, no backend.

## Files

- `index.html`: the page
- `assets/styles.css`: styles (light and dark mode)
- `assets/data.js`: all 43 lessons (notes, questions, model answers, sources)
- `assets/app.js`: routing, progress, review deck, search
- `assets/favicon.svg`: tab icon

## Serve it

Upload the folder as-is to any static host. It must be served over HTTP(S); opening `index.html` from disk also works but some browsers limit storage on `file://` pages.

- Local test: `python3 -m http.server 8000` in this folder, then open http://localhost:8000
- Netlify / Vercel / Cloudflare Pages: drag the folder in, or point a project at it with no build command and `.` as the output directory
- GitHub Pages: commit the folder contents to a repo and enable Pages for that branch
- nginx: point a `root` at the folder; no rewrites are needed because navigation uses `#` links

## How it differs from the claude.ai version

- Progress, answers, grades, review schedule and badges are saved in the visitor's browser (localStorage) only. They don't sync across devices or browsers, and clearing site data resets them.
- Each visitor has their own progress. Nothing is sent to a server.
- Fonts load from Google Fonts. Without internet access the page falls back to system fonts.

## Editing lessons

Lessons live in `assets/data.js` as one `DATA` array: modules contain `lessons`, and each lesson has `t` (title), `a` (notes), `n` (key points), `q` (question), `s` (model answer) and `r` (sources). Notes support `## ` headings, `- ` lists, `| ` tables (first row is the header), `! ` worked examples, `> ` takeaways, and `%%flow`, `%%seq` and `%%layers` diagram blocks.
