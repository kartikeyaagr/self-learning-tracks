# ML Maths Course: static web build

A single-page site. No build step, no server code, no database.

## Files
- `index.html`: the whole course (lessons, maths, diagrams, review deck). About 1.5 MB; the maths font is embedded.
- `step1_diagnostic.ipynb`: the diagnostic notebook, linked from the Diagnostic page.
- `favicon.svg`: tab icon.
- `.nojekyll`: stops GitHub Pages from running Jekyll. Harmless elsewhere.

## Run it locally
    cd ml-maths-course
    python3 -m http.server 8000
Then open http://localhost:8000. Opening `index.html` straight from disk also works.

## Put it on the web (pick one)
- **GitHub Pages:** push the folder's contents to a repository, then Settings > Pages > Deploy from branch > main, root folder.
- **Netlify:** drag the folder onto https://app.netlify.com/drop.
- **Cloudflare Pages:** Workers & Pages > Create > Pages > Upload assets, then upload the folder.
- **Any web server:** copy the files into its web root. Serve `.ipynb` as `application/x-ipynb+json` or `application/octet-stream` (most servers already do something sensible).

## What works differently from the Claude version
- **Progress is saved in the browser only** (localStorage key `mlmaths`). It does not sync between devices or browsers, and clearing site data erases it. In Claude it synced through the artifact's database.
- **"Report" on a source link** is kept in the same browser and not sent anywhere.
- **Notebook download** is a normal file download (no .zip).
- Fonts for the interface (Geist, IBM Plex Mono, Instrument Serif) load from Google Fonts. Without internet access the page falls back to system fonts. The maths font is embedded, so equations always render.
- Progress from the Claude version does not carry over.

## Updating
The page is generated from `course-src/` in the project files (`node build_ml.js`). Re-export after content changes rather than editing `index.html` by hand.
