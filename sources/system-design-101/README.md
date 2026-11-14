# System Design 101

A static, single-page course: 18 modules, 200 lessons, with diagrams, self-check questions and a flashcard review deck.

## What's in the folder

| Path | What it is |
|---|---|
| `index.html` | The whole site. All lesson data, styles and scripts are inside this one file. |
| `favicon.svg` | The site icon. |
| `src/` | Source files, needed only if you want to edit lessons and rebuild. |

## Serve it

No build step and no server code. Any static host works.

- **Try it locally:** `cd system-design-101 && python3 -m http.server 8000`, then open http://localhost:8000
- **Netlify:** drag the `system-design-101` folder onto app.netlify.com/drop
- **Cloudflare Pages / Vercel:** create a project from this folder. Leave the build command empty and set the output directory to `/`.
- **GitHub Pages:** push the folder's contents to a repo, then turn on Pages for the branch root.
- **nginx / Apache / S3:** upload `index.html` and `favicon.svg` to the web root.

The page loads three fonts from Google Fonts. Everything else is in `index.html`.

## Progress and notes

On a self-hosted copy, progress, written answers, grades, review schedule and streaks are saved in each visitor's browser (localStorage key `sdsd`). They stay on that browser and device, and clearing site data deletes them.

Progress you made on the Claude-hosted version does not carry over, because it was stored in that page's own database.

## Edit lessons and rebuild

Lesson notes are in `src/full/m01.txt` to `m18.txt`. Each lesson starts with `@NN.NN`. The notes format:

- `## Heading`, `- bullet`, `1. ordered item`, `| a | b |` table rows
- `! Example: ...` followed by more `! ` lines makes a worked example
- `> Takeaway: ...` makes the key takeaway, and `?? ...` is the model answer
- Diagrams go in blocks that end with a line containing only `%%`: `%%flow caption`, `%%seq caption` or `%%layers caption`

To rebuild (needs Python 3):

```
cd src
python3 build.py
```

This rewrites `index.html` in the folder above.

Lesson titles follow the fanout.sh system design roadmap. The notes are original summaries written from the sources linked in each lesson.
