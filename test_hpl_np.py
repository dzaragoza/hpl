"""Tests for hpl_np.py — stdlib unittest plus numpy, nothing else.

The benchmark path (gen_problem -> run_benchmark -> residual check)
is fully self-contained, and so are these tests: the repo needs only
hpl_np.py + requirements.txt + test_hpl_np.py to work on its own.
"""

import contextlib
import io
import unittest

import numpy as np

import hpl_np


class TestGenProblem(unittest.TestCase):

    def test_shapes_and_range(self):
        a, x_true = hpl_np.gen_problem(50, seed=42)
        self.assertEqual(a.shape, (50, 50))
        self.assertEqual(x_true.shape, (50,))
        self.assertTrue(np.all(a >= -1.0) and np.all(a < 1.0))
        self.assertTrue(np.all(x_true >= -1.0) and np.all(x_true < 1.0))

    def test_same_seed_same_problem(self):
        a1, x1 = hpl_np.gen_problem(30, seed=7)
        a2, x2 = hpl_np.gen_problem(30, seed=7)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(x1, x2)

    def test_different_seed_different_problem(self):
        a1, _ = hpl_np.gen_problem(30, seed=1)
        a2, _ = hpl_np.gen_problem(30, seed=2)
        self.assertFalse(np.array_equal(a1, a2))


class TestResidualChecks(unittest.TestCase):

    def test_exact_solution_gives_zero_residual(self):
        a, x_true = hpl_np.gen_problem(40, seed=3)
        b = a @ x_true
        scaled, fwd = hpl_np.residual_checks(a, b, x_true, x_true)
        self.assertEqual(scaled, 0.0)
        self.assertEqual(fwd, 0.0)

    def test_perturbed_solution_scores_worse(self):
        a, x_true = hpl_np.gen_problem(40, seed=3)
        b = a @ x_true
        x_bad = x_true + 1e-3
        good, fwd_good = hpl_np.residual_checks(a, b, x_true, x_true)
        bad, fwd_bad = hpl_np.residual_checks(a, b, x_bad, x_true)
        self.assertGreater(bad, good)
        self.assertGreater(fwd_bad, 0.0)

    def test_computed_residual_stays_small_at_any_n(self):
        """The point of the eps*||A||_1*||x||_1*N normalization: the
        backward error of a *computed* LU solution, measured in units of
        machine rounding, stays O(1) no matter how big N gets — which is
        why one threshold (16) is fair at any size."""
        for n in (100, 300):
            with self.subTest(n=n):
                a, x_true = hpl_np.gen_problem(n, seed=5)
                b = a @ x_true
                x = np.linalg.solve(a, b)
                scaled, _ = hpl_np.residual_checks(a, b, x, x_true)
                self.assertLess(scaled, 1.0)


class TestRunBenchmark(unittest.TestCase):

    def test_small_run_passes(self):
        res = hpl_np.run_benchmark(n=100, seed=1, repeats=2)
        self.assertEqual(res["n"], 100)
        self.assertEqual(res["seed"], 1)
        self.assertEqual(res["repeats"], 2)
        self.assertTrue(res["passed"])
        self.assertLess(res["scaled_residual"], 16.0)
        self.assertLess(res["forward_error"], 1e-8)
        self.assertGreater(res["gflops"], 0.0)

    def test_best_is_at_least_as_fast_as_first_and_median(self):
        """best time = min over all timed runs (first among them), and
        Gflop/s is inversely proportional to time — so best Gflop/s must
        beat or tie both the median and the first-call figures."""
        res = hpl_np.run_benchmark(n=150, seed=2, repeats=3)
        self.assertLessEqual(res["time"], res["first_time"])
        self.assertGreaterEqual(
            res["gflops"], res["median_gflops"] * (1 - 1e-12))
        self.assertGreaterEqual(
            res["gflops"], res["first_gflops"] * (1 - 1e-12))

    def test_gflops_uses_hpl_flop_count(self):
        res = hpl_np.run_benchmark(n=100, seed=3, repeats=1)
        flops = 2.0 / 3.0 * 100 ** 3 + 2.0 * 100 ** 2
        self.assertAlmostEqual(
            res["gflops"], flops / res["time"] / 1e9, places=6)

    def test_warmup_repeats_all_timed_runs(self):
        """repeats=k must produce exactly k timed runs: the loop runs
        1 + repeats iterations and skips only the warm-up."""
        res = hpl_np.run_benchmark(n=100, seed=4, repeats=4)
        self.assertEqual(res["repeats"], 4)

    def test_solution_actually_solves_the_system(self):
        a, x_true = hpl_np.gen_problem(60, seed=9)
        b = a @ x_true
        x = np.linalg.solve(a, b)
        self.assertAlmostEqual(
            float(np.abs(x - x_true).max()), 0.0, delta=1e-10)


class TestReportAndCLI(unittest.TestCase):

    def test_print_report_runs(self):
        res = hpl_np.run_benchmark(n=80, seed=6, repeats=1)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            hpl_np.print_report(res)
        out = buf.getvalue()
        self.assertIn("N=80", out)
        self.assertIn("PASSED", out)
        self.assertIn("Gflop/s", out)

    def test_bad_n_rejected(self):
        with self.assertRaises(SystemExit):
            hpl_np.main(["-n", "0"])

    def test_seed_option_accepted(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = hpl_np.main(["-n", "60", "--seed", "123", "--repeats",
                               "1"])
        self.assertEqual(code, 0)
        self.assertIn("seed=123", buf.getvalue())


if __name__ == "__main__":
    unittest.main()