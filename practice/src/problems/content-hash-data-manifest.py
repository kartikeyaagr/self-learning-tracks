import hashlib as _hl

META = dict(title='Content-addressed dataset manifest and upload plan', tags=['MLOps', 'Data Processing'], diff='Medium', runtime='python',
            lessons=['air-fanout-v1:14.02'], related=['hashed-dataset-split', 'perform-stable-exact-deduplication', 'model-registry-promotion'])

STATEMENT = r"""
A training run should record **which bytes** it read, not just which paths. Content-addressed storage names every file by the hash of its content, and a **manifest** lists the files of one dataset version together with the recipe that built it. Hashing the canonical manifest gives a single **dataset ID**.

Write `build_manifest(files, recipe, stored)`.

- `files`: a list of `[path, content]` pairs; `content` is a string, paths are unique.
- `recipe`: a dict of JSON-serialisable settings (filters, mixture weights, tokenizer version).
- `stored`: a list of SHA-256 hex digests of blobs the object store already holds.

## 1. Entries
For each file, encode `content` as UTF-8. Its `sha256` is `hashlib.sha256(data).hexdigest()` and its `bytes` is `len(data)` (bytes, not characters). An entry is the dict `{"path": ..., "sha256": ..., "bytes": ...}`. Sort the entries by path.

## 2. Dataset ID
Serialise `{"recipe": recipe, "files": entries}` with `json.dumps(..., sort_keys=True, separators=(",", ":"))`, encode it as UTF-8 and take the first 16 characters of its SHA-256 hex digest.

## 3. Upload plan
Content that the store already holds is never uploaded again, and identical content under two paths is uploaded once.
`upload` is the sorted list of **distinct** digests that are not in `stored`; `upload_bytes` is the total size of those distinct blobs.

Return `{"dataset_id": ..., "total_bytes": sum of bytes over all entries, "upload": [...], "upload_bytes": ...}`.
"""

STARTER = '''import hashlib
import json

def build_manifest(files, recipe, stored):
    """Return the dataset ID and the list of blobs to upload."""
    entries = [{"path": p, "sha256": hashlib.sha256(c.encode("utf-8")).hexdigest(), "bytes": len(c)}
               for p, c in files]
    canonical = json.dumps({"recipe": recipe, "files": entries})
    dataset_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    upload = [e["sha256"] for e in entries if e["sha256"] not in stored]
    return {"dataset_id": dataset_id, "total_bytes": sum(e["bytes"] for e in entries),
            "upload": upload, "upload_bytes": sum(e["bytes"] for e in entries if e["sha256"] in upload)}
'''

SOLUTION = '''import hashlib
import json

def build_manifest(files, recipe, stored):
    entries = []
    for path, content in files:
        data = content.encode("utf-8")
        entries.append({"path": path, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    entries.sort(key=lambda e: e["path"])                      # file order must not change the ID
    canonical = json.dumps({"recipe": recipe, "files": entries}, sort_keys=True, separators=(",", ":"))
    dataset_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    have = set(stored)
    new = {}                                                    # digest -> size, one per distinct blob
    for e in entries:
        if e["sha256"] not in have:
            new[e["sha256"]] = e["bytes"]
    return {"dataset_id": dataset_id, "total_bytes": sum(e["bytes"] for e in entries),
            "upload": sorted(new), "upload_bytes": sum(new.values())}
'''

HINTS = [
    'Two things must not change the dataset ID: the order of `files` and the order of keys inside `recipe`. Sort the entries by path and pass `sort_keys=True`.',
    '`len(content)` counts characters; storage counts bytes. Encode first, then measure.',
    'Collect the missing blobs in a dict keyed by digest, so a blob that appears under two paths is counted once.',
]


def _d(s):
    return _hl.sha256(s.encode('utf-8')).hexdigest()


_R1 = {"tokenizer": "bpe-64k-v2", "mixture": {"web": 0.75, "code": 0.25}, "dedup": "minhash-0.8"}
_R2 = {"tokenizer": "bpe-64k-v2", "mixture": {"web": 0.7, "code": 0.3}, "dedup": "minhash-0.8"}
_F1 = [["web/shard-000.txt", "the cat sat on the mat\n"], ["web/shard-001.txt", "a dog barked twice\n"],
       ["code/shard-000.txt", "def f(x):\n    return x + 1\n"]]
_F1_REV = list(reversed(_F1))
_F_CHANGED = [_F1[0], ["web/shard-001.txt", "a dog barked three times\n"], _F1[2]]
_F_DUP = _F1 + [["web/shard-002.txt", "the cat sat on the mat\n"], ["web/shard-003.txt", "fresh text\n"]]
_F_UNI = [["papers/p0.txt", "café résumé naïve\n"], ["papers/p1.txt", "plain ascii\n"]]
_S1 = [_d(c) for _, c in _F1]

TESTS = [
    (_F1, _R1, []),
    (_F1_REV, {"dedup": "minhash-0.8", "mixture": {"code": 0.25, "web": 0.75}, "tokenizer": "bpe-64k-v2"}, []),
    (_F_CHANGED, _R1, _S1),
    (_F_DUP, _R1, []),
    (_F_DUP, _R1, _S1[:1]),
    (_F1, _R2, _S1),
    (_F_UNI, {}, []),
    ([], _R1, _S1),
]

EXPLANATION = r"""
## Why hash content, then hash the manifest
A path can be rewritten; a digest cannot. Naming blobs by their SHA-256 makes storage deduplicating (identical content is one object) and verifiable (rehash after download). Hashing the canonical manifest gives the whole dataset one ID that changes whenever any file **or the recipe** changes, and only then. That ID is what a run record stores.

## Canonical serialisation
`json.dumps` keeps dict insertion order by default and adds spaces after separators. Two semantically identical manifests can then hash differently. Sorting keys and entries and fixing separators makes the bytes, and so the ID, depend only on the content. Test 2 has the files reversed and the recipe keys reordered, and gets the same ID as test 1.

## The upload plan
Tests 3 to 5 are the everyday case: a new version shares most blobs with the last one, so only changed content moves. Test 5 changes only the mixture weights: nothing is uploaded, but the dataset ID changes, because a different mixture trains a different model.

## Complexity
Hashing is linear in the total bytes; sorting is $O(n \log n)$ in the number of files. Real tools hash large files in chunks and cache digests by file size and modification time to avoid rehashing unchanged files.

## Interview follow-ups
- Why do tools such as DVC store a directory as its own hashed listing of child hashes? (It is this manifest, one level down: a Merkle tree.)
- How would you make the ID independent of how the data is split into shards?
- What breaks if the tokenizer version is left out of the recipe?
"""
