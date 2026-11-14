# Self-learning tracks

Courses and practice for machine learning, AI and the systems behind them, as one static site:
<https://kartikeyaagr.github.io/self-learning-tracks/>

- **Courses**: ML Maths, ML Toolkit, Machine Learning, Deep Learning, AI Research, AI Engineering, Inference
  Engineering and System Design, on a layered roadmap.
- **Practice Lab**: 700+ coding problems with tests, hints and solutions. Python, NumPy, Pandas and SQL run in the
  browser; PyTorch, JAX, CUDA and Triton open in Colab. Study plans, paper implementations, projects, timed interview rounds.
- **One progress store** (in the browser), one spaced-review deck, full-text search.

## Layout

| Path | What |
|---|---|
| `site/` | The Astro site. Lessons are Markdown in `site/src/content/courses/<course>/lessons/`, and they are the source of truth |
| `practice/` | Practice Lab sources (`src/problems/*.py`, plans, quizzes, rounds) and the build that verifies them into `site/public/practice/` |
| `sources/` | Inputs the course importer converts: claude.ai export bundles, add-on modules, JSON course drafts |
| `scripts/` | `import_courses.py` (sources → Markdown), `extract_courses.py` / `extract_diagrams.py` (each course's figure code and CSS) |

## Develop

    cd site && npm ci && npm run dev          # http://localhost:4321/self-learning-tracks/
    cd site && npm run build                  # static site + search index in site/dist
    uv run practice/src/build.py              # after changing practice problems

Pushing to `main` deploys with GitHub Actions (`.github/workflows/deploy.yml`).
