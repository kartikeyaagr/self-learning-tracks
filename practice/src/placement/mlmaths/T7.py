TITLE = 'Estimates on paper'
MODULES = ['10']

STATEMENT = r'''
Fill in `ANSWERS` with numbers. Use 1 GB = $10^9$ bytes. Count a multiply-add as 2 FLOPs. Work them out on paper; the check only looks at the numbers (tolerance 10 to 30%).

- **a.** Memory in GB of a float32 matrix with $n = 2{,}000{,}000$ rows and $d = 1024$ columns.
- **b.** FLOPs to form $X^\top X$ for that matrix (no symmetry tricks).
- **c.** Seconds for (b) on a T4 GPU in fp32 (peak 8.1 TFLOP/s), assuming 50% of peak.
- **d.** Memory in GB to store the full Hessian of a model with $10^6$ parameters in float32.
- **e.** FLOPs for one training step (forward plus backward) of an MLP $784 \to 1024 \to 1024 \to 10$ at batch size 256. Ignore biases and activations; assume backward costs 2x forward.
- **f.** GB of memory for weights, gradients and optimiser state when training a 7B-parameter model with Adam in mixed precision: bf16 weights and gradients, plus fp32 master weights and fp32 Adam moments. Ignore activations.
- **g.** Monte Carlo samples needed to estimate a probability near $p = 0.01$ with standard error $0.001$.
- **h.** An alert fires for 99% of real incidents and for 1% of non-incidents. 0.1% of time windows contain a real incident. Probability that a given alert is a real incident?
'''

STARTER = '''ANSWERS = {
    "a": None,  # GB
    "b": None,  # FLOPs
    "c": None,  # seconds
    "d": None,  # GB
    "e": None,  # FLOPs
    "f": None,  # GB
    "g": None,  # samples
    "h": None,  # probability
}
'''

CHECK = r'''_KEY = json.loads(base64.b64decode("eyJBIjogeyJhIjogOC4xOTIsICJiIjogNDE5NDAwMDAwMDAwMC4wLCAiYyI6IDEuMDM2LCAiZCI6IDQwMDAuMCwgImUiOiAyODYwMDAwMDAwLjAsICJmIjogMTEyLjAsICJnIjogOTkwMC4wLCAiaCI6IDAuMDkwMn0sICJUIjogeyJhIjogMC4xLCAiYiI6IDAuMTUsICJjIjogMC4zLCAiZCI6IDAuMSwgImUiOiAwLjE1LCAiZiI6IDAuMSwgImciOiAwLjEsICJoIjogMC4xfX0="))
def _check():
    missing = [k for k, v in ANSWERS.items() if v is None]
    if len(missing) == len(_KEY["A"]):
        raise NotImplementedError
    expect(not missing, f"missing answers: {missing}")
    wrong = []
    for k, ref in _KEY["A"].items():
        v = float(ANSWERS[k])
        if not abs(v - ref) <= _KEY["T"][k] * abs(ref):
            wrong.append(k)
    expect(not wrong, f"outside tolerance: {wrong}. Recheck units and factors of 2.")
'''

SOLUTION = '''ANSWERS = {
    "a": 8.192,     # 2e6 * 1024 * 4 bytes = 8.19e9 B
    "b": 4.194e12,  # n * d^2 multiply-adds * 2 = 2e6 * 1024^2 * 2
    "c": 1.036,     # 4.194e12 / (0.5 * 8.1e12)
    "d": 4000.0,    # (1e6)^2 entries * 4 bytes = 4e12 B
    "e": 2.86e9,    # forward 2 * 256 * (784*1024 + 1024*1024 + 1024*10) = 9.53e8; times 3 with backward
    "f": 112.0,     # 7e9 * (2 bf16 weights + 2 bf16 grads + 4 master + 4 + 4 Adam moments) = 7e9 * 16 B
    "g": 9900.0,    # SE = sqrt(p(1-p)/n)  =>  n = 0.01 * 0.99 / 0.001^2
    "h": 0.0902,    # 0.99 * 0.001 / (0.99 * 0.001 + 0.01 * 0.999)
}
'''

MCQ = [
    {"q": r"You store the matrix from (a) in bf16 instead of float32 and form $X^\top X$ the same way. How do (a) and (b) change?",
     "options": [r"(a) is unchanged, because memory depends only on the shape; (b) halves.",
                 r"(a) halves to about 4.1 GB; (b) is unchanged at about $4.2 \times 10^{12}$ FLOPs.",
                 r"(a) and (b) both halve: bf16 numbers are half the size, so each operation counts as half a FLOP.",
                 r"(a) drops to a quarter, since bf16 keeps far fewer mantissa bits; (b) is unchanged."],
     "answer": 1,
     "why": r"Bytes per entry go from 4 to 2, while the FLOP count depends only on the shapes. Only the time in (c) can change, through the hardware's throughput for that format."},
    {"q": "For alert (h), which single change raises the probability that an alert is real the most?",
     "options": ["Halving the miss rate (99% to 99.5% detection): the answer roughly doubles to about 0.18.",
                 "Both raise it by the same amount, since each halves one of the two error rates.",
                 "Neither helps much; only raising the base rate of incidents can change the answer.",
                 "Halving the false-positive rate (1% to 0.5%): the answer rises from about 0.09 to about 0.17. Halving the miss rate barely moves it (about 0.091)."],
     "answer": 3,
     "why": "With a 0.1% base rate, false alarms dominate the denominator: $0.99 \\times 0.001 = 0.00099$ true alerts against $0.01 \\times 0.999 \\approx 0.00999$ false ones. Halving the false alarms gives $0.00099/(0.00099 + 0.004995) \\approx 0.165$."},
]
