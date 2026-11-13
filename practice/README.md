# Practice Lab

Coding problems with tests, hints, full solutions and explanations. Python, NumPy, Pandas and SQL run in the
browser (Pyodide in a Web Worker). PyTorch, JAX, CUDA (Numba kernels) and Triton problems open as Colab notebooks.

## Build

    uv run practice/src/build.py        # verify every problem, write index.html, data/, nb/
    uv run --no-project build_hub.py    # refresh the hub card

## Add a problem

Create `src/problems/<slug>.py` with:

| Name | What |
|---|---|
| `META` | `title`, `tags` (from `PARTS` in build.py; first tag = topic), `diff` (Easy/Medium/Hard), `runtime` (python, numpy, pandas, sql, torch), optional `lessons` (`"trackkey:lessonid"`), `related` (slugs), `examples` (default 2), `tol` |
| `STATEMENT` | Markdown with `$math$` |
| `STARTER` | Code the learner starts from. It must fail at least one test |
| `SOLUTION` | Reference code. Expected outputs are computed from it |
| `HINTS` | At least 2, from a nudge to a near-giveaway |
| `TESTS` | At least 5 tuples of arguments (for SQL: setup scripts). The first `examples` are shown |
| `EXPLANATION` | Why it works, complexity, common bugs, interview follow-ups |
| `REFERENCE` | Triton only: code defining the same function with PyTorch on CPU. Expected outputs come from it, because the build machine has no GPU; SOLUTION is syntax-checked only |

The build fails if the solution doesn't pass its own tests twice with identical output, if the starter passes,
or if a tag, lesson or related slug doesn't exist. `harness.py` grades both here and in the browser.

## Files

- `index.html`, `data/*.json`, `nb/*.ipynb`: generated, commit them (GitHub Pages serves them).
- `runner.js`: the Web Worker. Pyodide is pinned to 0.29.5; the build pins Python 3.13, numpy 2.2 and pandas 2.3 to match.
- Progress is stored in `localStorage` under `practice`, in the same shape as the courses, so the hub reads it.
