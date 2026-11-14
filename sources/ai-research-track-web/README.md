# AI Research Track: web export

A static site. It needs no build step and no server code.

## Contents

- `index.html`: the whole course (notes, figures, quizzes and the review deck), in one file.
- `sheets/`: the 21 Colab notebooks that the lesson pages link to.
- `_headers`: cache and security headers for Cloudflare Pages or Netlify. Other hosts ignore it.

## Deploy on Cloudflare Pages (no Git needed)

1. Go to Cloudflare dashboard → Workers & Pages → Create → Pages → Upload assets.
2. Name the project, then drag in this folder (or `ai-research-track-web.zip`).
3. Click Deploy. The site goes live at `https://<project>.pages.dev`.
4. To update it later, upload the new folder as a new deployment of the same project.

From the command line instead: `npx wrangler pages deploy . --project-name ai-research-track`

## Other hosts

- **Netlify:** drag the folder onto app.netlify.com/drop.
- **GitHub Pages:** push the files to a repository and enable Pages on the main branch, root folder.
- **Local test:** run `python3 -m http.server 8000` in this folder, then open http://localhost:8000.

## How it differs from the claude.ai version

- **Progress is saved in your browser only** (localStorage key `air-fanout-v1`). It does not sync between devices or with the claude.ai copy, and clearing site data erases it.
- **Notebooks are direct downloads** from `sheets/`. In Colab, use File → Upload notebook.
- **One-click "Open in Colab"** works only from GitHub. If you host the files in a public GitHub repository, a notebook opens at `https://colab.research.google.com/github/<user>/<repo>/blob/main/sheets/<file>.ipynb`.
- **Fonts and maths:** fonts come from Google Fonts. The maths is pre-rendered into the page, so it needs no other network access.

## Privacy

The site has no analytics and no backend. Anyone with the URL can read it. Use Cloudflare Access to restrict it if you want it private.
