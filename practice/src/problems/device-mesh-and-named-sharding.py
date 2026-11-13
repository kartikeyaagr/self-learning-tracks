import numpy as np

META = dict(title='Where does each shard live? Meshes and PartitionSpecs', tags=['Distributed Computing', 'JAX'],
            diff='Easy', runtime='jax', lessons=['ieng:18.01', 'ieng:18.03'],
            related=['data-parallel-step-with-psum', 'tensor-parallel-mlp'])

STATEMENT = r"""
In JAX, devices are arranged in a named **mesh** (for example 4 × 2 with axes `('data', 'model')`), and a `PartitionSpec` says which mesh axes split which array dimension. `NamedSharding(mesh, P('data', None))` splits dimension 0 four ways and replicates it across `'model'`. Work out what each device holds, with plain Python: no devices needed.

Write `shard_info(global_shape, mesh_names, mesh_shape, spec, coord)`:

- `global_shape`: list of ints, the array's shape.
- `mesh_names`, `mesh_shape`: lists, the mesh axes in order and their sizes.
- `spec`: list with one entry per array dimension (it may be shorter than the rank; missing trailing entries mean `None`). Each entry is `None` (not split), an axis name, or a list of axis names (split over their product; the **first** name is the most significant).
- `coord`: list of ints, one device's coordinate in the mesh (same order as `mesh_names`).

Return a dict:

- `'shard_shape'`: list, the local shape on every device;
- `'replicas'`: int, how many devices hold an identical copy of each shard (the product of the sizes of mesh axes the spec does not use);
- `'slices'`: list of `[start, stop]` per dimension, the part of the global array held by the device at `coord`.

For a dimension split over axes $a_1, \dots, a_m$, the shard index is the mixed-radix number $(c_{a_1}, \dots, c_{a_m})$, so with two axes it is $c_{a_1} \cdot |a_2| + c_{a_2}$.

Return `None` if the spec is invalid: an unknown axis name, an axis used more than once, more entries than dimensions, or a dimension not divisible by its number of shards.

This problem runs in **Colab** with the JAX runtime, but the function itself is plain Python. The notebook includes the same tests.
"""

STARTER = '''def shard_info(global_shape, mesh_names, mesh_shape, spec, coord):
    """Return {'shard_shape': [...], 'replicas': int, 'slices': [[start, stop], ...]} or None."""
    raise NotImplementedError
'''

SOLUTION = '''import math

def shard_info(global_shape, mesh_names, mesh_shape, spec, coord):
    size = dict(zip(mesh_names, mesh_shape))
    pos = dict(zip(mesh_names, coord))
    if len(spec) > len(global_shape):
        return None
    spec = list(spec) + [None] * (len(global_shape) - len(spec))

    used, shard_shape, slices = [], [], []
    for dim, entry in zip(global_shape, spec):
        axes = [] if entry is None else [entry] if isinstance(entry, str) else list(entry)
        if any(a not in size for a in axes):
            return None
        used += axes
        n = math.prod(size[a] for a in axes)          # number of shards along this dimension (1 if replicated)
        if dim % n:
            return None
        idx = 0
        for a in axes:                                # mixed radix, first axis most significant
            idx = idx * size[a] + pos[a]
        shard = dim // n
        shard_shape.append(shard)
        slices.append([idx * shard, (idx + 1) * shard])

    if len(used) != len(set(used)):                   # a mesh axis can split only one dimension, once
        return None
    replicas = math.prod(s for a, s in size.items() if a not in used)
    return {'shard_shape': shard_shape, 'replicas': replicas, 'slices': slices}
'''

HINTS = [
    'Normalise each spec entry to a list of axis names (`None` -> `[]`, `"data"` -> `["data"]`). The number of shards along a dimension is the product of those axes\' sizes.',
    'The device\'s shard index along a dimension combines its mesh coordinates on those axes in order: `idx = idx * size[a] + coord[a]`. Its slice is `[idx * shard, (idx + 1) * shard]`.',
    'Replicas are the product of the sizes of the mesh axes that appear nowhere in the spec. Check for repeated or unknown axes and for divisibility before returning.',
]

TESTS = [
    ([8, 6], ['data', 'model'], [4, 2], ['data', None], [2, 1]),
    ([8, 6], ['data', 'model'], [4, 2], ['data', 'model'], [3, 0]),
    ([16, 6], ['data', 'model'], [4, 2], [['data', 'model'], None], [1, 1]),
    ([16, 6], ['data', 'model'], [4, 2], [['model', 'data'], None], [1, 1]),
    ([5, 4], ['x'], [2], ['x'], [0]),
    ([32, 64], ['data', 'model'], [2, 4], [None, 'model'], [1, 3]),
    ([8, 8], ['data', 'model'], [2, 2], ['data', 'data'], [0, 0]),
    ([4, 4, 12], ['pp', 'dp', 'tp'], [2, 2, 3], [], [1, 0, 2]),
    ([4, 4, 12], ['pp', 'dp', 'tp'], [2, 2, 3], ['dp', None, 'tp'], [1, 1, 2]),
    ([8], ['data'], [4], ['expert'], [0]),
]

EXPLANATION = r"""
## The mental model
A sharding is a map from device to a rectangular slice of the global array. Mesh axes named in the spec cut the array; mesh axes not named replicate it. In test 1, dimension 0 is cut 4 ways along `'data'`, so the device at `(2, 1)` holds rows 4 to 6 of all columns, and the two `'model'` devices in each row of the mesh hold identical copies (`replicas = 2`). You can check shard shapes against JAX without any devices: `NamedSharding(AbstractMesh((4, 2), ('data', 'model')), P('data', None)).shard_shape((8, 6))`.

## Order of a multi-axis split
`P(('data', 'model'))` and `P(('model', 'data'))` both cut a dimension 8 ways but assign the pieces to devices differently (tests 3 and 4: shard 3 versus shard 5 for the same device). The order matters when it must line up with another array's sharding, or with which devices share fast links.

## Typical specs
- Data parallel: batch split over `'data'`, parameters replicated (`P()`).
- Tensor parallel: weight matrices split over `'model'` on the hidden dimension.
- FSDP: parameters also split over `'data'`, gathered just before use.

## Common bugs
- Using one mesh axis for two dimensions (test 7): each device would need two different pieces of the same axis.
- Indivisible dimensions (test 5): `jax.device_put` raises; pad the array or change the mesh.
- Forgetting that missing trailing entries mean replicated.

## Interview follow-ups
- On 8 devices, how would you lay out a mesh for 2-way tensor and 4-way data parallelism if devices 0 to 3 share NVLink?
- What does XLA insert when two operands of a matmul have incompatible shardings? (Collectives: all-gather, reduce-scatter or all-to-all, chosen by the partitioner.)
"""
