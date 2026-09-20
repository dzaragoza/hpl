# HPL-Py — the Linpack benchmark in readable Python

A re-implementation of the pipeline behind the
[HPL](https://www.netlib.org/benchmark/hpl/) benchmark — the program that
decides the TOP500 ranking of the world's fastest supercomputers — in a
single readable file that hands the heavy lifting to LAPACK through NumPy.

The real HPL is ~20,000 lines of Fortran/C tuned for cache blocking, MPI
distribution, and vendor BLAS kernels.  All of that complexity exists to
execute one rather small algorithm fast.  This project is that algorithm,
naked — with the fast path being exactly what real HPL calls at the bottom
of its stack: LAPACK's `dgetrf` (LU with partial pivoting) and `dgetrs`
(triangular solves), inside a threaded, vectorized BLAS.

## A note on provenance

All code in this repository was written by an AI coding agent
(Vibe Code, built by Mistral AI) at the direction of the repository
owner, who reviewed, tested, and tuned it.  It is shared in the spirit
of the project itself: making something normally opaque — the Linpack
benchmark, and now its authorship too — a little more readable.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt                      # just numpy>=1.24

python3 hpl_np.py                    # N=4096
./hpl_np.py -n 4096 --repeats 3      # same thing: the script is executable
python3 hpl_np.py --blas-info       # which BLAS? how many cores are used?
python3 -m unittest test_hpl_np     # 31 tests
```

Under the **fish** shell, `source`/`deactivate` are bash-isms; use fish's
`activate` wrapper instead:

```fish
python3 -m venv .venv
source .venv/bin/activate.fish
pip install -r requirements.txt

./hpl_np.py                          # N=4096
./hpl_np.py -n 4096 --repeats 3
./hpl_np.py --blas-info
python3 -m unittest test_hpl_np

deactivate                           # fish's own, leaves the venv
```

## What it does

The same five stages as real HPL, one function each:

1. **Generate** a random `N×N` matrix `A` and a known solution `x_true`,
   then form the right-hand side `b = A·x_true`.
2. **Factor** `A` into `P·A = L·U` — LU with partial pivoting.
3. **Solve** `L·U·x = P·b` by two triangular sweeps.
4. **Check** the answer with the HPL residual:

   ```
   scaled_residual = ||A·x − b||_∞ / (eps · ||A||_1 · ||x||_1 · N)
   ```

   HPL accepts anything below 16.  Because it is measured in units of
   machine rounding (`eps` ~ 2.2e-16), the *same* threshold is fair at
   N=300 on a laptop and at N=10⁷ on a supercomputer.
5. **Time** steps 2–3 and report

   ```
   R [Gflop/s] = (2/3·N³ + 2·N²) / time
   ```

In code, steps 2–3 are one call — `np.linalg.solve(A, b)` — running
inside OpenBLAS/MKL: SIMD-vectorized, cache-blocked, and multi-threaded
(BLAS releases the GIL, so every core is used with no multiprocessing on
your side).

## Measuring like a benchmark, not a script

Timing follows real benchmark practice (HPL's own): one **warm-up** run
first — BLAS spins up its thread pool, caches fill, the CPU leaves its
low-power state — then the **best** of `--repeats` timed runs is the
headline Rmax-style number.  First-call and median figures are printed
too, because the gap between them is itself informative.

## Sizing N for your machine

`np.linalg.solve` does not destroy your input, so it works on a full
copy of `A`: peak memory is ~2× the matrix, i.e. ~2×8·N² bytes.  For a
32 GiB laptop, that means N ≈ 40,000 is the practical ceiling — roughly
`N ≈ √(0.35 × RAM / 8)` in general.  Past that, the run swaps and the
"Gflop/s" number becomes disk speed, not compute.

Expect performance to rise with N and then flatten (more parallelism,
then memory-bandwidth saturation).  One laptop's ladder, for
calibration (AMD Ryzen 7 7840U, 8 cores, pip-installed numpy/OpenBLAS):

| N | Gflop/s |
|---|---|
| 8192 | 64 |
| 16384 | 113 |
| 32768 | 181 |

181 Gflop/s is roughly the #1 TOP500 machine of late 1995 — a useful
reminder of what 30 years of cache-blocking and SIMD engineering bought.

## When would this machine have been a supercomputer?

Every run ends with a TOP500 calibration.  `top500_data.json` (13 KB,
next to the script) holds one record per list edition since June 1993:
the Rmax of that edition's #1 system and of its #500.  After the report,
the benchmark answers the fun question — the **last** edition your
Gflop/s would have topped, and the **last** one you would have made at
all:

```
TOP500 calibration              (68 editions, June 1993 - June 2026)
------------------------------------------------------------------------------
     181.0 Gflop/s would have been:
  the world's No. 1 in November 1995   (then No. 1: Numerical Wind
Tunnel, ... at 170.0 Gflop/s)
  still on the list in June 2002   (entry threshold then: 134.3 Gflop/s)
```

That is the whole fun of it: the same 32-GiB laptop was the world's
fastest machine in 1995 — and off the bottom of the list by 2002.

Machines that are too slow for even the very first (June 1993, entry
0.422 Gflop/s) list get a friendly note instead.  No network access
happens at run time; `--no-top500` skips the section.  The data file was
extracted from the top500.org list archives — refresh it after each
June/November edition with `python3 top500_update.py`.

## The serial-BLAS trap

pip numpy wheels bundle **threaded OpenBLAS**; a distro-packaged numpy
may instead be linked against the serial reference BLAS — silently
~20× slower, one core pegged while the others idle.  If your numbers look
single-threaded, run:

```bash
python3 hpl_np.py --blas-info
```

It prints which BLAS numpy is linked against, the thread-cap environment
variables, and measures the CPU-time/wall-time ratio of a big dgemm —
i.e. how many cores the BLAS *actually* uses.  Fix: install numpy from
pip inside a venv (or switch your distro's BLAS alternatives to
OpenBLAS).

## Tuning beyond N

- **Threads:** hyperthreads share FPUs, so 8 threads often beat 16 on a
  GEMM-heavy run.  Sweep with
  `OPENBLAS_NUM_THREADS=8 python3 hpl_np.py -n 16384 --repeats 3`.
- **NB (panel size)** is real HPL's second-most-important knob — inside
  LAPACK it is chosen heuristically (`ILAENV`) and is not exposed here.
- Real HPL also tunes its process grid (P×Q), look-ahead depth, and
  broadcast shape in `HPL.dat`; those belong to the distributed algorithm
  and have no equivalent in a single `solve` call.

## Files

- `hpl_np.py` — the benchmark (docstrings double as the explainer)
- `test_hpl_np.py` — 31 tests: generator, residual-check semantics,
  benchmark invariants, flop count, report, CLI, TOP500 lookup
- `top500_data.json` — all 68 TOP500 editions (June 1993 - June 2026):
  #1 and #500 Rmax per edition, in Gflop/s
- `top500_update.py` — maintenance: re-download the lists into the JSON
- `requirements.txt` — numpy>=1.24
- `LICENSE` — MIT
- `README.md` — this file

## License

[MIT](LICENSE) — use it, fork it, teach with it.
