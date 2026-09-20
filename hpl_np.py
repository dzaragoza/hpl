"""NumPy version of the tiny HPL Linpack benchmark.

This is what "real life" Python Linpack looks like: the classic HPL
pipeline (generate -> solve -> residual check -> Gflop/s), with the
factorization and solve delegated to LAPACK through NumPy.

    A x = b   ->   x = np.linalg.solve(A, b)

That one call runs LAPACK's dgetrf (LU with partial pivoting) and
dgetrs (triangular solves) — the same routines real HPL calls — inside
OpenBLAS/MKL, which are:

    * written in C/Fortran with vectorized (SIMD) kernels,
    * cache-blocked (the trailing update is a tiled dgemm, so data is
      reused from L2/L3 instead of streaming from DRAM),
    * internally multi-threaded (BLAS releases the GIL, so it uses
      every core without any multiprocessing on your side).

For calibration: a pure-Python implementation of the same algorithm runs
at ~0.017 Gflop/s — the gap you will see here, typically four orders of
magnitude, IS the point: it is what the 20,000 tuned lines of HPL and the
vendor BLAS beneath them are buying.

Note on residuals: the HPL-style check here is computed in NumPy too,
which itself uses pairwise summation — so the residual you get can be
slightly better than a naive summation would give at the same N.

Run it:

    python3 hpl_np.py                    # N=2000
    python3 hpl_np.py -n 4000
    python3 hpl_np.py -n 4096 --repeats 3
    python3 hpl_np.py --blas-info        # is my BLAS using all my cores?

Installation (if numpy is missing):

    pip install -r requirements.txt
"""

import argparse
import sys
import time

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip install -r requirements.txt "
             "(or: pip install numpy)")

EPS = sys.float_info.epsilon


def gen_problem(n, seed):
    """Random A (N x N) and known solution x_true, entries in [-1, 1)."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 1.0, size=(n, n))
    x_true = rng.uniform(-1.0, 1.0, size=n)
    return a, x_true


def residual_checks(a, b, x, x_true):
    """HPL's pass/fail check, in the same spirit as the classic check.

    scaled_residual = ||A x - b||_inf / (eps * ||A||_1 * ||x||_1 * N)

    HPL accepts anything below 16: the backward error measured in units
    of machine rounding, so it is fair at any N.  (We also have x_true
    on hand, so the forward error is reported too, for curiosity.)
    """
    n = len(b)
    r = a @ x - b
    rinf = np.abs(r).max()
    norm1_a = np.abs(a).sum(axis=0).max()
    norm1_x = np.abs(x).sum()
    scaled = rinf / (EPS * norm1_a * norm1_x * n)
    fwd_err = np.abs(x - x_true).max()
    return scaled, fwd_err


def run_benchmark(n, seed, repeats=3):
    """One NumPy benchmark run: warm-up, best-of-k, residual check.

    Real benchmark practice (and HPL's own): warm up first, then take the
    BEST of several timed runs.  The first call is always slower — BLAS
    spins up its thread pool, caches are cold, the CPU may still be in a
    low-power state — none of which is the machine's sustained rate.
    The best-of-k number is the honest Rmax-style figure; the first-call
    number is what a one-shot script would actually feel.
    """
    a, x_true = gen_problem(n, seed)
    b = a @ x_true

    times = []
    for i in range(1 + repeats):          # 1 warm-up + repeats timed
        t0 = time.perf_counter()
        x = np.linalg.solve(a, b)         # LAPACK dgetrf + dgetrs, threaded
        elapsed = time.perf_counter() - t0
        if i > 0:
            times.append(elapsed)

    best = min(times)
    median = sorted(times)[len(times) // 2]
    scaled, fwd_err = residual_checks(a, b, x, x_true)
    flops = 2.0 / 3.0 * n ** 3 + 2.0 * n ** 2      # HPL's operation count
    return {
        "n": n,
        "seed": seed,
        "repeats": repeats,
        "time": best,
        "first_time": times[0],
        "gflops": flops / best / 1e9,
        "median_gflops": flops / median / 1e9,
        "first_gflops": flops / times[0] / 1e9,
        "scaled_residual": scaled,
        "forward_error": fwd_err,
        "passed": scaled < 16.0,
    }


def blas_info():
    """Diagnose the BLAS threading situation: which library, what caps,
    and how many cores it ACTUALLY uses on a big dgemm.

    The core-count measurement is a stdlib trick: time.process_time()
    counts CPU time across ALL threads of the process, while
    perf_counter() counts wall time.  CPU/wall during a matmul is
    therefore ~the number of cores the BLAS put to work:

        ~1.0  -> single-threaded BLAS (or a thread cap)
        ~8.0  -> 8 cores busy

    (OpenBLAS threads spin-wait between tiles, which slightly inflates
    the ratio when threads idle -- but "roughly N" is exactly what we
    want to know here.)
    """
    import os
    print("=" * 78)
    print("BLAS diagnostics")
    print("-" * 78)
    print("numpy version                 : {}".format(np.__version__))
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        val = os.environ.get(var)
        print("{:28s} : {}".format(var,
                                     val if val is not None else "<unset>"))
    print("-" * 78)
    print("BLAS that numpy is linked against (np.show_config):" + "\n")
    np.show_config()

    print("-" * 78)
    n = 3000
    a = np.random.default_rng(0).uniform(-1.0, 1.0, (n, n))
    reps = 3
    a @ a                                  # warm-up: thread pool, caches
    t0 = time.perf_counter()
    c0 = time.process_time()
    for _ in range(reps):
        a @ a                              # dgemm: 2*n^3 flops each
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0
    gflops = reps * 2.0 * n ** 3 / wall / 1e9
    print("dgemm benchmark                : {:10.1f} Gflop/s".format(gflops))
    print("CPU-time / wall-time           : {:10.2f}"
          "   <- roughly, cores actually used".format(cpu / wall))
    print("=" * 78)
    if cpu / wall < 1.5:
        print("VERDICT: BLAS is effectively SINGLE-THREADED.  If a thread")
        print("cap variable above is set, unset it (before starting Python).")
        print("If not, your numpy is linked against a serial BLAS — see")
        print("the np.show_config output above; 'openblas' is the threaded")
        print("one pip wheels bundle, 'blas' (netlib) is the slow serial one.)")
    else:
        print("VERDICT: BLAS IS using multiple cores.  If hpl_np.py still")
        print("feels one-cored, your N is probably too small to parallelize")
        print("— try -n 4000 and watch the CPU monitor again.")


def print_report(res):
    verdict = "PASSED" if res["passed"] else "FAILED"
    print("=" * 78)
    print("HPL-Py NumPy (LAPACK)         N={}   seed={}".format(
        res["n"], res["seed"]))
    print("-" * 78)
    print("Time for factor + solve        : {:10.4f} s"
          "   (best of {} after warm-up)".format(
              res["time"], res["repeats"]))
    print("Performance                    : {:10.4f} Gflop/s"
          "   (best; median {:.1f}, first {:.1f})".format(
              res["gflops"], res["median_gflops"], res["first_gflops"]))
    print("-" * 78)
    print("||Ax-b||_inf/(eps*||A||_1*||x||_1*N) = {:.3e}   {}".format(
        res["scaled_residual"], verdict))
    print("forward error   ||x-x_true||_inf                   = {:.3e}".format(
        res["forward_error"]))
    print("=" * 78)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="NumPy HPL Linpack benchmark (np.linalg.solve)")
    parser.add_argument("-n", "--n", type=int, default=2000,
                        help="matrix order N (default 2000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default 42)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="timed runs after the warm-up (default 3); "
                             "the best one is reported")
    parser.add_argument("--blas-info", action="store_true",
                        help="diagnose BLAS threading (library, caps, "
                             "cores actually used) and exit")
    args = parser.parse_args(argv)
    if args.blas_info:
        blas_info()
        return 0
    if args.n < 1:
        parser.error("N must be positive")

    res = run_benchmark(args.n, args.seed, args.repeats)
    print_report(res)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())