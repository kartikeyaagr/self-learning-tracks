META = dict(title='GPU utilization and idle gaps from a kernel timeline', tags=['GPU Programming', 'Inference'],
            diff='Medium', runtime='python', lessons=['ieng:12.03', 'ieng:12.06'],
            related=['simulate-continuous-decode-batching', 'route-cache-aware-prefill-requests'])

STATEMENT = r"""
A profiler trace (such as Nsight Systems or the PyTorch profiler) shows each GPU kernel as an interval on a stream. The first questions you ask of it: what fraction of the time is the GPU doing anything, where are the idle gaps (often CPU launch overhead or synchronisation), and how much do kernels on different streams actually overlap?

Write `timeline_stats(kernels, window_start, window_end, gap_threshold)` and return a dict.

## Inputs
- `kernels`: list of `(stream, start, end)` with an int stream id and float times in microseconds, `start < end`. Kernels may be in any order and may overlap, also on the same stream.
- `window_start < window_end`: the analysis window. Clip every kernel to it; a kernel entirely outside the window contributes nothing.
- `gap_threshold`: float. Report idle gaps strictly longer than this.

## Output
| key | value |
|---|---|
| `"busy"` | float, length of the union of all clipped intervals |
| `"utilization"` | float, `busy / (window_end - window_start)` |
| `"per_stream"` | dict `{stream: busy fraction of the window for that stream alone}`, for every stream that appears in `kernels` (0.0 if all its kernels fall outside) |
| `"overlap"` | float, total time during which **at least two** kernels are running (on any streams) |
| `"gaps"` | list of `(start, end)` tuples: maximal idle intervals inside the window (including before the first and after the last kernel) longer than `gap_threshold`, in time order |

Intervals are half-open, $[\text{start}, \text{end})$: a kernel ending at $t$ and another starting at $t$ leave no gap and don't overlap.
"""

STARTER = '''def timeline_stats(kernels, window_start, window_end, gap_threshold):
    """Return a dict: busy, utilization, per_stream, overlap, gaps."""
    raise NotImplementedError
'''

SOLUTION = '''def _sweep(intervals, lo, hi):
    """Clip to [lo, hi), then return (time with >=1 active, time with >=2 active, idle gaps)."""
    events = []
    for s, e in intervals:
        s, e = max(s, lo), min(e, hi)
        if s < e:
            events += [(s, 1), (e, -1)]
    events.sort(key=lambda ev: (ev[0], ev[1]))      # at equal times, ends (-1) before starts (+1)
    busy = overlap = 0.0
    gaps, active, t_prev = [], 0, lo
    for t, d in events:
        if active >= 1:
            busy += t - t_prev
        if active >= 2:
            overlap += t - t_prev
        if active == 0 and t > t_prev:
            gaps.append((t_prev, t))
        active += d
        t_prev = t
    if t_prev < hi:                                   # trailing idle time (active is 0 here)
        gaps.append((t_prev, hi))
    return busy, overlap, gaps

def timeline_stats(kernels, window_start, window_end, gap_threshold):
    span = window_end - window_start
    busy, overlap, gaps = _sweep([(s, e) for _, s, e in kernels], window_start, window_end)
    per_stream = {}
    for sid in sorted({k[0] for k in kernels}):
        b, _, _ = _sweep([(s, e) for st, s, e in kernels if st == sid], window_start, window_end)
        per_stream[sid] = b / span
    return {"busy": busy, "utilization": busy / span, "per_stream": per_stream, "overlap": overlap,
            "gaps": [g for g in gaps if g[1] - g[0] > gap_threshold]}
'''

HINTS = [
    'Turn every clipped interval into two events, `(start, +1)` and `(end, -1)`, sort them, and sweep while counting how many kernels are active.',
    'Between consecutive event times, add the elapsed time to `busy` if at least one kernel is active and to `overlap` if at least two are; if none is active, that stretch is a gap. Sort ends before starts at equal times so touching kernels leave no zero-length gap.',
    'Gaps also include the stretch from `window_start` to the first event and from the last event to `window_end`. Reuse the same sweep per stream for `per_stream`.',
]

TESTS = [
    ([(0, 0.0, 10.0), (0, 20.0, 30.0)], 0.0, 40.0, 5.0),
    ([(0, 0.0, 10.0), (1, 5.0, 15.0), (0, 15.0, 20.0)], 0.0, 20.0, 0.0),
    ([(1, 12.0, 18.0), (0, 2.0, 6.0), (0, 4.0, 9.0), (2, 30.0, 31.0)], 5.0, 25.0, 2.0),
    ([(0, -5.0, 3.0), (0, 50.0, 60.0), (3, 7.0, 7.5)], 0.0, 10.0, 0.25),
    ([(0, float(i), i + 0.75) for i in range(8)], 0.0, 8.0, 0.2),
    ([(0, 0.0, 100.0), (1, 10.0, 20.0), (2, 15.0, 30.0), (1, 25.0, 26.0)], 0.0, 100.0, 1.0),
    ([(i % 3, 10.0 * i + (i % 2) * 2.5, 10.0 * i + 6.0 + (i % 4) * 2.0) for i in range(12)], 0.0, 130.0, 3.0),
    ([(5, 1.0, 2.0)], 3.0, 4.0, 0.5),
]

EXPLANATION = r"""
## Why a sweep line
Summing kernel durations double-counts overlapping kernels, and simply looking at the first start and last end hides the gaps in between. Sorting the $2n$ start and end events and walking them in order with an "active" counter answers all three questions at once: time with active $\ge 1$ is busy, active $\ge 2$ is overlap, active $= 0$ is idle. Half-open intervals plus "ends before starts" at equal timestamps make back-to-back kernels (test 5's 0.25 µs gaps, test 2's touching kernels at 15) behave as expected.

## Reading the numbers
- **Low utilization with many small gaps** (test 5): typical of decode with small batches, where each kernel is shorter than the CPU's launch latency. CUDA graphs and kernel fusion remove the gaps.
- **Overlap**: compute on one stream overlapping copies or communication on another is what you want in a well-pipelined trace. Note that overlapping kernels may still be slowed by competing for the same SMs or bandwidth, so overlap time is an upper bound on real concurrency.
- **Per stream** fractions can add up to more than `utilization` when streams overlap (test 6).

## Complexity
$O(n \log n)$ for the sort; the sweep itself is linear.

## Common bugs
- Not clipping to the window before computing the union, or dropping kernels that straddle a window edge (test 4).
- Treating a kernel that ends exactly when the next starts as leaving a zero-length gap.
- Reporting gaps only between kernels and missing the leading and trailing idle time (test 3).

## Interview follow-ups
- GPU utilization reported by `nvidia-smi` is "a kernel was running", not "the SMs were busy". What metric would you use to see whether those kernels actually use the hardware?
- Your trace shows 30% idle time between decode steps. List three likely causes and how you would confirm each one.
"""
