#!/usr/bin/env python3
"""NumPy version of the tiny HPL Linpack benchmark.

NOTE: this file was written by an AI coding agent (Vibe Code,
Mistral AI) at the direction of the repository owner, who reviewed
and tested it.  See the README for details.

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

Problem sizing: users think in memory, not matrix order, so the main
knob is -g GIB — the size of the N x N matrix (8 bytes per entry,
N = floor(sqrt(GIB * 2^30 / 8))).  The default is 25% of the detected
physical RAM, which is also the safe side of the ~2x peak during the
solve (np.linalg.solve works on a copy of A), so a default run tops
out around 50% of RAM.  HPL traditionalists can still set N directly
with -n; the report echoes both the GiB and the N.  RAM detection is
standard library only: /proc/meminfo on Linux, GlobalMemoryStatusEx
on Windows, sysctl on macOS, sysconf() elsewhere; if all of that
fails, N=4096 (a 0.12 GiB matrix — safe on anything that runs numpy).

While it runs, each phase is announced — generate, warm-up, timed
runs with the Gflop/s so far, residual check — because a benchmark
that sits silent for minutes looks exactly like a crash.

Run it:

    python3 hpl_np.py                    # auto: 25% of your RAM
    python3 hpl_np.py -g 4               # a 4 GiB matrix
    python3 hpl_np.py -n 4000            # or set N the HPL way
    python3 hpl_np.py -g 4 --repeats 3
    python3 hpl_np.py --blas-info        # is my BLAS using all my cores?

If top500_data.json (all TOP500 list editions, built by
top500_update.py) sits next to this script, each run ends with the fun
question: when would this machine have been a supercomputer — the last
edition it would have topped, and the last it would have made at all.
No network access happens during a run.

Installation (if numpy is missing):

    pip install -r requirements.txt
"""

import argparse
import json
import os
import sys
import time

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip install -r requirements.txt "
             "(or: pip install numpy)")

EPS = sys.float_info.epsilon
TOP500_FILE = "top500_data.json"
RAM_FRACTION = 0.25
FALLBACK_N = 4096


def physical_ram_bytes():
    """Best-effort total physical RAM of this machine, or None.

    Used only to pick a safe default problem size.  Linux reads
    /proc/meminfo, Windows the GlobalMemoryStatusEx API through ctypes,
    macOS the sysctl behind 'hw.memsize', and other POSIX systems
    sysconf() — every route is standard library.
    """
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    if sys.platform == "win32":
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_uint),
                        ("dwMemoryLoad", ctypes.c_uint),
                        ("ullTotalPhys", ctypes.c_uint64),
                        ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64),
                        ("ullAvailPageFile", ctypes.c_uint64),
                        ("ullTotalVirtual", ctypes.c_uint64),
                        ("ullAvailVirtual", ctypes.c_uint64),
                        ("ullAvailExtendedVirtual", ctypes.c_uint64)]
        stat = MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullTotalPhys)
    if sys.platform == "darwin":
        import subprocess
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
            return int(out.strip())
        except (OSError, ValueError):
            pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def n_to_gib(n):
    """Memory size of the N x N matrix, in GiB (8 bytes per entry)."""
    return n * n * 8.0 / 2 ** 30


def gib_to_n(gib):
    """The largest matrix order N whose matrix fits in gib GiB."""
    return int((gib * 2 ** 30 / 8.0) ** 0.5)


def choose_size(n=None, gib=None):
    """Turn the user's size request into (N, GiB, RAM bytes).

    An explicit -n wins, then an explicit -g, then the default: 25% of
    the detected physical RAM for the matrix — which peaks around 50%
    of RAM during the solve, because np.linalg.solve factors a copy of
    A.  If RAM cannot be detected at all, N=4096, small enough to be
    safe on any machine that can run numpy at all.
    """
    ram = physical_ram_bytes()
    if n is not None:
        return n, n_to_gib(n), ram
    if gib is not None:
        return gib_to_n(gib), gib, ram
    if ram:
        gib = RAM_FRACTION * ram / 2 ** 30
        return gib_to_n(gib), gib, ram
    return FALLBACK_N, n_to_gib(FALLBACK_N), None


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


def run_benchmark(n, seed, repeats=3, progress=None):
    """One NumPy benchmark run: warm-up, best-of-k, residual check.

    Real benchmark practice (and HPL's own): warm up first, then take the
    BEST of several timed runs.  The first call is always slower — BLAS
    spins up its thread pool, caches are cold, the CPU may still be in a
    low-power state — none of which is the machine's sustained rate.
    The best-of-k number is the honest Rmax-style figure; the first-call
    number is what a one-shot script would actually feel.

    progress, if given, is called with a message string at each phase
    and after every run — a big problem otherwise sits silent for
    minutes, and a silent benchmark looks like a crash.
    """
    if progress is None:
        progress = lambda msg: None
    flops = 2.0 / 3.0 * n ** 3 + 2.0 * n ** 2      # HPL's operation count
    progress("Phase 1/3: generating the random {}x{} problem...".format(n, n))
    a, x_true = gen_problem(n, seed)
    b = a @ x_true
    progress("Phase 2/3: solving  A x = b  (1 warm-up + {} timed run{})".format(
        repeats, "" if repeats == 1 else "s"))

    times = []
    for i in range(1 + repeats):          # 1 warm-up + repeats timed
        t0 = time.perf_counter()
        x = np.linalg.solve(a, b)         # LAPACK dgetrf + dgetrs, threaded
        elapsed = time.perf_counter() - t0
        if i == 0:
            progress("  warm-up : {:8.2f} s   (not counted)".format(elapsed))
        else:
            times.append(elapsed)
            progress("  run {}/{}: {:8.2f} s   -> {:8.1f} Gflop/s"
                     "   (best so far: {:8.1f})".format(
                         i, repeats, elapsed, flops / elapsed / 1e9,
                         flops / min(times) / 1e9))

    progress("Phase 3/3: checking the residual (HPL's pass/fail test)...")
    best = min(times)
    median = sorted(times)[len(times) // 2]
    scaled, fwd_err = residual_checks(a, b, x, x_true)
    return {
        "n": n,
        "gib": n_to_gib(n),
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
        print("feels one-cored, your matrix is probably too small to")
        print("parallelize — try -g 2 or bigger and watch the CPU monitor")
        print("again.")


def load_top500(path=None):
    """Read the bundled TOP500 history, or None if it is unusable.

    top500_data.json sits next to this script and holds one record per
    list edition since June 1993: the Rmax of the #1 system (what it
    took to top the list) and of the #500 system (what it took to get
    ON it at all).  It was generated by top500_update.py from the
    top500.org list archives; hpl_np.py itself never touches the
    network — a benchmark that downloads things mid-run would be
    timing its own web requests.

    A missing or corrupt file is not an error: the calibration is a
    bonus, so we just skip it.
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            TOP500_FILE)
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    raw = data.get("editions") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return None
    editions = []
    for ed in raw:
        try:
            if ed["rmax_top"] > 0 and ed["rmax_entry"] > 0 and ed["edition"]:
                editions.append(ed)
        except (KeyError, TypeError, ValueError):
            continue
    if not editions:
        return None
    editions.sort(key=lambda e: e["edition"])
    return editions


def top500_calibration(gflops, editions):
    """The fun question: when would this machine have been a
    supercomputer?

    Walks the editions (June 1993 to the present, twice a year) and
    returns (top, entry):

        top    — the LAST edition whose No. 1 your Gflop/s matches or
                  beats: the last time you would have been the fastest
                  machine on Earth
        entry  — the LAST edition whose No. 500 you match or beat: the
                  last time you would have appeared on the list at all

    Either can be None (never fast enough for that honour).  Note the
    asymmetry: entry thresholds grew ~1000x faster than the No. 1, so a
    typical laptop tops some 1990s list but drops off the bottom of the
    list only a few years later.
    """
    top = None
    entry = None
    for ed in editions:
        if gflops >= ed["rmax_top"]:
            top = ed
        if gflops >= ed["rmax_entry"]:
            entry = ed
    return top, entry


def print_top500(res):
    """Print the "when was I a supercomputer?" section of the report."""
    editions = load_top500()
    if editions is None:
        return
    gflops = res["gflops"]
    top, entry = top500_calibration(gflops, editions)
    print()
    print("=" * 78)
    print("TOP500 calibration              ({} editions, {} - {})".format(
        len(editions), editions[0]["label"], editions[-1]["label"]))
    print("-" * 78)
    if entry is None:
        first = editions[0]
        print("{:10.1f} Gflop/s would never have made ANY TOP500 list".format(
            gflops))
        print("— not even the very first one ({}), whose weakest listed".format(
            first["label"]))
        print("machine ran {:.3f} Gflop/s.  Don't feel bad: that machine".format(
            first["rmax_entry"]))
        print("cost millions in 1993.  Linpack is a big-N game; give the")
        print("benchmark a larger N (and a threaded BLAS) and come back.")
    else:
        print("{:10.1f} Gflop/s would have been:".format(gflops))
        if top is not None:
            print("  the world's No. 1 in {}   (then No. 1: {}, at "
                  "{:.1f} Gflop/s)".format(
                      top["label"], top["top_system"], top["rmax_top"]))
        else:
            print("  on the list, but never No. 1 — every edition's No. 1 "
              "was faster")
        print("  still on the list in {}   (entry threshold then: "
              "{:.1f} Gflop/s)".format(
                  entry["label"], entry["rmax_entry"]))
        print("A laptop saying \"I was a supercomputer once\" is the "
          "whole point")
        print("of 30 years of Moore's law — enjoy it :)")
    print("=" * 78)


def print_report(res):
    verdict = "PASSED" if res["passed"] else "FAILED"
    print("=" * 78)
    print("HPL-Py NumPy (LAPACK)         N={}   seed={}".format(
        res["n"], res["seed"]))
    print("-" * 78)
    print("Matrix size (N x N, 8 B each)  : {:10.2f} GiB".format(
        res["gib"]))
    print("Peak RAM during the solve      : ~{:9.2f} GiB"
          "   (LAPACK factors a copy of A)".format(2.0 * res["gib"]))
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
    parser.add_argument("-g", "--gib", type=float, default=None,
                        help="size of the N x N matrix in GiB (default: "
                             "25%% of the machine's physical RAM, a safe "
                             "choice for a ~50%% peak during the solve)")
    parser.add_argument("-n", "--n", type=int, default=None,
                        help="matrix order N directly — the HPL "
                             "traditionalist's spelling of --gib")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default 42)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="timed runs after the warm-up (default 3); "
                             "the best one is reported")
    parser.add_argument("--blas-info", action="store_true",
                        help="diagnose BLAS threading (library, caps, "
                            "cores actually used) and exit")
    parser.add_argument("--no-top500", action="store_true",
                        help="skip the end-of-run TOP500 calibration")
    args = parser.parse_args(argv)
    if args.blas_info:
        blas_info()
        return 0
    if args.n is not None and args.gib is not None:
        parser.error("give either -n or -g, not both")
    if args.n is not None and args.n < 1:
        parser.error("N must be positive")
    if args.gib is not None and args.gib <= 0:
        parser.error("--gib must be positive")

    n, gib, ram = choose_size(args.n, args.gib)
    auto = args.n is None and args.gib is None

    def note(msg):
        print(msg, flush=True)

    print("=" * 78)
    if auto and ram:
        print("Sizing: {:.1f} GiB of RAM detected -> default matrix is".format(
            ram / 2 ** 30))
        print("25% of that: {:.2f} GiB -> N={}".format(gib, n))
    elif auto:
        print("Sizing: could not detect RAM -> safe default N={}".format(n))
        print("(a {:.2f} GiB matrix; pass -g GIB to choose a size)".format(gib))
    else:
        print("Sizing: {:.2f} GiB matrix -> N={}".format(gib, n))
    print("The solve briefly needs ~2x the matrix: ~{:.2f} GiB of RAM.".format(
        2.0 * gib))
    print("=" * 78)

    res = run_benchmark(n, args.seed, args.repeats, progress=note)
    print_report(res)
    if not args.no_top500:
        print_top500(res)
    print()
    print("Tip: this was a {:.2f} GiB matrix (N={}) — Gflop/s keeps".format(
        gib, n))
    print("rising with size as the BLAS finds more parallel work.  Try")
    print("-g 8, -g 16, ... (a run needs ~2x the matrix size in RAM, so")
    print("stay under ~50% of your machine's memory.)")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
