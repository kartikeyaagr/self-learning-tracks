META = dict(title='Back-of-the-envelope QPS, storage, bandwidth and fleet size', tags=['Distributed Computing'],
            diff='Easy', runtime='python', lessons=['sdsd:01.06'],
            related=['plan-kv-cache-capacity', 'optimize-throughput-and-cost-under-latency-slo', 'training-compute-budget'])

STATEMENT = r"""
Turn a product's assumptions into the capacity numbers a design needs. Write `capacity_plan(a)` where `a` is a
dict of assumptions:

| key | meaning |
| --- | --- |
| `dau` | daily active users |
| `writes_per_user` | writes per active user per day (may be a decimal such as `0.2`) |
| `read_ratio` | reads per write (`100` means a 100:1 read:write ratio; `0` for a write-only service) |
| `peak_factor` | peak traffic divided by average traffic (may be a decimal such as `2.5`) |
| `payload_bytes` | bytes stored per write, and bytes returned per read |
| `retention_years` | how long writes are kept |
| `replication` | number of stored copies |
| `server_qps` | requests per second one API server handles at acceptable latency |
| `utilization_pct` | the integer percentage of `server_qps` you plan to run at (headroom) |

## Units (use these exactly, not the 10^5 s/day shortcut)
- 1 day = 86,400 s; 1 year = 365 days.
- Decimal units: 1 MB = $10^6$ bytes, 1 TB = $10^{12}$ bytes, 1 Gbit = $10^9$ bits, 1 byte = 8 bits.

## Return a dict with these keys
- `write_qps`, `read_qps`: average requests per second ($\text{writes per day} = \text{dau} \cdot \text{writes\_per\_user}$,
  reads per day = writes per day $\cdot$ `read_ratio`).
- `peak_write_qps`, `peak_read_qps`: the averages times `peak_factor`.
- `storage_tb`: everything written over the retention period, times `replication`, in TB.
- `peak_ingress_mb_s`: bytes per second written at peak, in MB/s.
- `peak_egress_gbit_s`: bytes per second read at peak, in **Gbit/s** (network links are sold in bits).
- `servers`: the smallest integer number of API servers whose combined planned capacity
  ($\text{server\_qps} \cdot \text{utilization\_pct}/100$ each) covers peak reads plus peak writes.

All values except `servers` are floats (checked to a relative tolerance of $10^{-6}$). `servers` is an `int`
and must be exact: when peak traffic is an exact multiple of one server's planned capacity, do not round up.
Treat decimal inputs as the exact decimals they are written as (`fractions.Fraction(str(x))`), because
$1500 \cdot 1.1$ in floats is `1650.0000000000002`.
"""

STARTER = '''def capacity_plan(a):
    """Return a dict of capacity numbers for the assumptions in `a`."""
    writes = a["dau"] * a["writes_per_user"]
    w_qps = writes / 1e5                          # "a day is about 10^5 seconds"
    r_qps = w_qps * a["read_ratio"]
    pw, pr = w_qps * a["peak_factor"], r_qps * a["peak_factor"]
    return {
        "write_qps": w_qps, "read_qps": r_qps,
        "peak_write_qps": pw, "peak_read_qps": pr,
        "storage_tb": writes * a["payload_bytes"] * 365 * a["retention_years"] / 1e12,
        "peak_ingress_mb_s": pw * a["payload_bytes"] / 1e6,
        "peak_egress_gbit_s": pr * a["payload_bytes"] / 1e9,
        "servers": round((pw + pr) / a["server_qps"]),
    }
'''

SOLUTION = '''from fractions import Fraction
from math import ceil

DAY = 86_400

def capacity_plan(a):
    x = lambda k: Fraction(str(a[k]))              # exact decimals: no 1650.0000000000002
    writes = x("dau") * x("writes_per_user")       # per day
    w_qps = writes / DAY
    r_qps = w_qps * x("read_ratio")
    pw, pr = w_qps * x("peak_factor"), r_qps * x("peak_factor")
    payload = x("payload_bytes")
    stored = writes * payload * 365 * x("retention_years") * x("replication")
    per_server = x("server_qps") * x("utilization_pct") / 100
    return {
        "write_qps": float(w_qps), "read_qps": float(r_qps),
        "peak_write_qps": float(pw), "peak_read_qps": float(pr),
        "storage_tb": float(stored / 10**12),
        "peak_ingress_mb_s": float(pw * payload / 10**6),
        "peak_egress_gbit_s": float(pr * payload * 8 / 10**9),   # bytes -> bits
        "servers": ceil((pw + pr) / per_server),                 # exact ceil on a Fraction
    }
'''

HINTS = [
    'Chain the units: users -> writes per day -> divide by 86,400 for average QPS -> multiply by the peak factor. Reads are writes times `read_ratio`.',
    'Storage counts only writes: writes per day x bytes x 365 x years x replicas. Bandwidth is peak QPS x bytes; multiply by 8 for bits.',
    'Servers must handle peak reads plus writes at the planned utilisation, so divide by `server_qps * utilization_pct / 100` and take the ceiling. Do it on `Fraction`s so an exact fit is not pushed up by a float error.',
]

TESTS = [
    ({"dau": 100_000_000, "writes_per_user": 10, "read_ratio": 0, "peak_factor": 3, "payload_bytes": 1000,
      "retention_years": 1, "replication": 3, "server_qps": 2000, "utilization_pct": 60},),
    ({"dau": 150_000_000, "writes_per_user": 2, "read_ratio": 100, "peak_factor": 2, "payload_bytes": 1000,
      "retention_years": 5, "replication": 3, "server_qps": 5000, "utilization_pct": 60},),
    ({"dau": 50_000_000, "writes_per_user": 0.2, "read_ratio": 10, "peak_factor": 3, "payload_bytes": 2_100_000,
      "retention_years": 1, "replication": 3, "server_qps": 1000, "utilization_pct": 70},),
    ({"dau": 864_000, "writes_per_user": 10, "read_ratio": 4, "peak_factor": 2.5, "payload_bytes": 500,
      "retention_years": 2, "replication": 3, "server_qps": 500, "utilization_pct": 50},),
    ({"dau": 12_960_000, "writes_per_user": 10, "read_ratio": 0, "peak_factor": 1.1, "payload_bytes": 2000,
      "retention_years": 3, "replication": 2, "server_qps": 550, "utilization_pct": 60},),
    ({"dau": 2_000_000, "writes_per_user": 1, "read_ratio": 20, "peak_factor": 50, "payload_bytes": 4000,
      "retention_years": 1, "replication": 3, "server_qps": 3000, "utilization_pct": 75},),
    ({"dau": 1000, "writes_per_user": 3, "read_ratio": 5, "peak_factor": 2, "payload_bytes": 300,
      "retention_years": 10, "replication": 1, "server_qps": 1000, "utilization_pct": 100},),
]

EXPLANATION = r"""
## The chain
Every number comes from one chain of units: users x actions per day = events per day; divide by 86,400 s for
the average rate; multiply by the peak factor for what you must actually survive; divide by what one box
does at your planned utilisation for the fleet. Storage is the same events times bytes, times days kept,
times copies.

Test 1 is the lesson's question: $10^8 \cdot 10 = 10^9$ writes a day is about 11,600 per second on average and
34,700 at a 3x peak. $10^9 \cdot 1\text{ KB} = 1$ TB a day, 365 TB a year, about 1.1 PB with 3 replicas. At
2,000 requests per server run at 60%, that is 29 servers; the more important conclusion is that 35k writes
per second is far beyond one relational primary, so writes need partitioning.

## Why exact units here
In an interview you round hard ($86{,}400 \approx 10^5$) and say so. In code that other people reuse, the 15%
error of the shortcut becomes a wrong fleet size, so the tests use 86,400. The other trap is the ceiling: test
4 needs exactly $1250 / 250 = 5$ servers and test 5 (a write-only ingest service) exactly $1650 / 330 = 5$.
Computed in floats, $1500 \cdot 1.1 = 1650.0000000000002$ and the ceiling buys a 6th server. Rounding to the nearest integer
instead of up (the starter's bug) under-provisions whenever the fraction is below one half.

## Reading the numbers
- Test 2 (150M users, 100:1 reads): about 6,900 writes but 694,000 reads per second at peak. Reads dominate,
  which argues for caching and precomputed timelines, not a bigger database.
- Test 3 (photos): only 116 uploads per second on average, yet 729 MB/s of peak ingress, 58 Gbit/s of peak
  egress and 23 PB a year with 3 copies. Storage and egress cost, not request rate, drive the design: object storage and a CDN.
- Test 6 (ticket drop, 50x peak): an average of 486 requests per second becomes 24,300 at peak and 11 servers
  where the average needs one. Event-driven products are sized for the spike.

## Common mistakes
- Mixing bits and bytes: a 10 Gbit/s link moves 1.25 GB/s.
- Forgetting replication and the read path's egress.
- Sizing the fleet for average traffic, or at 100% utilisation, so one zone failing takes you down.

## Interview follow-ups
- Add 30% for indexes and metadata and 20% yearly growth: how does storage change over 5 years?
- Which of these numbers changes the architecture, and which only changes the bill?
"""
