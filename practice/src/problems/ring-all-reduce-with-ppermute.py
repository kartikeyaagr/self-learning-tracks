import numpy as np

META = dict(title='Ring all-reduce from ppermute', tags=['Distributed Computing', 'JAX'], diff='Hard', runtime='jax',
            lessons=['ieng:18.04', 'ieng:18.02'],
            related=['data-parallel-step-with-psum', 'tensor-parallel-mlp', 'device-mesh-and-named-sharding'], tol=[1e-5, 1e-6])

STATEMENT = r"""
`psum` hides the algorithm. The classic one for large buffers is the **ring all-reduce**: $N$ devices in a ring, each buffer cut into $N$ chunks, a **reduce-scatter** phase that leaves each device with one fully summed chunk, then an **all-gather** phase that circulates the summed chunks. Each phase is $N - 1$ steps of `jax.lax.ppermute` (send to your right neighbour, receive from your left).

Write `ring_all_reduce(x)`:

- `x`: `(N, L)` float32; row $i$ is device $i$'s buffer, and $N$ divides $L$. Chunk $c$ is `buf[c * L/N : (c + 1) * L/N]`.
- Simulate the devices with `jax.vmap(per_device, axis_name='ring')`. Use `jax.lax.axis_index('ring')` for a device's id $i$ and `jax.lax.ppermute(v, 'ring', perm=[(j, (j + 1) % N) for j in range(N)])` to shift data one step right.

Schedule, for $s = 0, \dots, N - 2$:

1. **Reduce-scatter step** $s$: device $i$ sends its chunk $(i - s) \bmod N$; it receives chunk $(i - 1 - s) \bmod N$ from device $i - 1$ and **adds** it to its own copy of that chunk.

After the last reduce-scatter step, device $i$ holds the full sum of chunk $(i + 1) \bmod N$.

2. **All-gather step** $s$: device $i$ sends its chunk $(i + 1 - s) \bmod N$; it receives chunk $(i - s) \bmod N$ and **overwrites** its copy.

Return `(reduced, after_scatter)`, both `(N, L)` float32: every device's buffer at the end (all rows equal to the column sums of `x`), and every device's buffer right after the reduce-scatter phase (partially reduced chunks included). Do not call `psum`.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def ring_all_reduce(x):
    """x (N, L): row i is device i's buffer. Return (reduced (N, L), after_scatter (N, L))."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def ring_all_reduce(x):
    x = jnp.asarray(x, jnp.float32)
    N, L = x.shape
    right = [(j, (j + 1) % N) for j in range(N)]

    def per_device(buf):
        i = jax.lax.axis_index('ring')
        chunks = buf.reshape(N, L // N)                       # chunk c is row c
        for s in range(N - 1):                                # N is static, so this unrolls N - 1 steps
            recv = jax.lax.ppermute(chunks[(i - s) % N], 'ring', perm=right)
            chunks = chunks.at[(i - 1 - s) % N].add(recv)     # reduce: accumulate the neighbour's partial sum
        scattered = chunks.reshape(L)
        for s in range(N - 1):
            recv = jax.lax.ppermute(chunks[(i + 1 - s) % N], 'ring', perm=right)
            chunks = chunks.at[(i - s) % N].set(recv)         # gather: copy the finished chunk
        return chunks.reshape(L), scattered

    return jax.vmap(per_device, axis_name='ring')(x)
'''

HINTS = [
    'Reshape each buffer to `(N, L // N)` so chunk `c` is row `c`. The device id `i = jax.lax.axis_index("ring")` is a traced value, but `chunks[(i - s) % N]` and `chunks.at[...].add(...)` work with traced indices.',
    'In each step every device sends at the same time: one `ppermute` with the "shift right" permutation delivers device `i - 1`\'s chunk to device `i`. `N` is a Python int, so a Python `for` loop over steps is fine.',
    'Reduce-scatter adds the received chunk into chunk `(i - 1 - s) % N`; all-gather overwrites chunk `(i - s) % N` with the received one, which device `i - 1` sent from index `((i - 1) + 1 - s) % N`.',
]

_r = np.random.default_rng(36)
TESTS = [
    (np.array([[1.0, 2.0], [10.0, 20.0]]),),
    (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),),
    (np.arange(9.0).reshape(3, 3).repeat(2, axis=1),),
    (_r.integers(-5, 6, size=(4, 8)).astype(float),),
    (np.ones((1, 3)),),
    (_r.normal(size=(5, 10)),),
    (_r.integers(0, 100, size=(8, 16)).astype(float),),
]

EXPLANATION = r"""
## Tracing the reduce-scatter
Follow chunk 0 with $N = 3$. At step 0, device 0 sends it to device 1, which now holds $x_0 + x_1$ for chunk 0. At step 1, device 1 sends that to device 2, which adds $x_2$: device 2 owns the full sum of chunk $0 = (2 + 1) \bmod 3$. Every chunk travels the ring the same way, starting from a different device, so after $N - 1$ steps each device owns one finished chunk. The all-gather then rotates the finished chunks around the ring, overwriting stale copies.

## Why rings are bandwidth-optimal
Each device sends $2 (N - 1)$ chunks of size $L / N$, about $2L$ numbers in total, **independent of $N$**. A naive scheme (everyone sends everything to one root) puts $N L$ on the root's link. The cost is latency: $2(N - 1)$ sequential steps, which is why large clusters use trees or hierarchical rings (within a node over NVLink, across nodes over the network) for small messages.

## Correctness checks
`after_scatter` exposes the intermediate state: device $i$'s chunk $(i + 1) \bmod N$ equals the column sums of that chunk, while its other chunks hold partial sums. With $N = 1$ (test 5) both phases are empty and the buffer is returned unchanged. Integer-valued inputs make the sums exact regardless of the order of additions.

## Common bugs
- Sending and receiving the same chunk index: the receiver must update the chunk its **left** neighbour sent, which is one index lower.
- Using `set` in the reduce phase or `add` in the gather phase.
- Running the loop over a traced $N$: the number of steps must be static.

## Interview follow-ups
- How does reduce-scatter alone give you ZeRO-style sharded gradients, and why is all-reduce = reduce-scatter + all-gather useful to remember?
- Why can the floating-point result of an all-reduce differ between ring sizes, and when does that matter?
"""
