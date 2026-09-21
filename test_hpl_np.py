"""Tests for hpl_np.py — stdlib unittest plus numpy, nothing else.

NOTE: this file was written by an AI coding agent (Vibe Code,
Mistral AI) at the direction of the repository owner, who reviewed
and tested it.  See the README for details.

The benchmark path (gen_problem -> run_benchmark -> residual check)
is fully self-contained, and so are these tests: the repo needs only
hpl_np.py + requirements.txt + test_hpl_np.py to work on its own.

The TOP500 tests additionally use top500_data.json when it sits next
to hpl_np.py (regenerate it with top500_update.py); they self-skip
otherwise, and the pure lookup logic is tested on a built-in mini list.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest

import numpy as np

import hpl_np


class TestTop500(unittest.TestCase):
    """The end-of-run calibration against the bundled list history."""

    MINI = [
        {"edition": "1993-06", "label": "June 1993", "rmax_top": 59.7,
         "rmax_entry": 0.422, "top_system": "CM-5/1024, Los Alamos"},
        {"edition": "1997-06", "label": "June 1997", "rmax_top": 1068.0,
         "rmax_entry": 5.7, "top_system": "ASCI Red, Sandia"},
        {"edition": "2002-06", "label": "June 2002", "rmax_top": 35860.0,
         "rmax_entry": 105.8, "top_system": "Earth Simulator, JAMSTEC"},
        {"edition": "2020-06", "label": "June 2020", "rmax_top": 415530000.0,
         "rmax_entry": 1340000.0, "top_system": "Fugaku, RIKEN"},
    ]

    def test_would_have_topped_1993(self):
        top, entry = hpl_np.top500_calibration(100.0, self.MINI)
        self.assertIsNotNone(top)
        self.assertEqual(top["edition"], "1993-06")
        self.assertEqual(entry["edition"], "1997-06")

    def test_never_number_one_but_on_the_list(self):
        top, entry = hpl_np.top500_calibration(10.0, self.MINI)
        self.assertIsNone(top)
        self.assertEqual(entry["edition"], "1997-06")

    def test_boundary_exact_match_counts(self):
        top, _ = hpl_np.top500_calibration(1068.0, self.MINI)
        self.assertEqual(top["edition"], "1997-06")

    def test_too_slow_for_any_list(self):
        top, entry = hpl_np.top500_calibration(0.1, self.MINI)
        self.assertIsNone(top)
        self.assertIsNone(entry)

    def test_faster_than_every_edition(self):
        top, entry = hpl_np.top500_calibration(1e9, self.MINI)
        self.assertEqual(top["edition"], "2020-06")
        self.assertEqual(entry["edition"], "2020-06")

    def test_calibration_is_monotone(self):
        for g in (0.1, 1.0, 60.0, 1068.0, 35861.0, 5e8):
            top, entry = hpl_np.top500_calibration(g, self.MINI)
            if top is not None:
                self.assertLessEqual(top["rmax_top"], g)
            if entry is not None:
                self.assertLessEqual(entry["rmax_entry"], g)

    def test_bundled_data_is_usable_and_sorted(self):
        editions = hpl_np.load_top500()
        if editions is None:
            self.skipTest("top500_data.json not next to hpl_np.py")
        self.assertGreaterEqual(len(editions), 60)
        self.assertEqual([e["edition"] for e in editions],
                         sorted(e["edition"] for e in editions))
        for ed in editions:
            self.assertGreater(ed["rmax_top"], ed["rmax_entry"])
        self.assertEqual(editions[0]["edition"], "1993-06")
        self.assertLess(editions[0]["rmax_entry"], 1.0)

    def test_load_top500_missing_file_returns_none(self):
        self.assertIsNone(hpl_np.load_top500("/nonexistent/top500.json"))

    def test_load_top500_corrupt_file_returns_none(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as fh:
            fh.write("{not json at all")
            path = fh.name
        self.addCleanup(os.unlink, path)
        self.assertIsNone(hpl_np.load_top500(path))

    def test_load_top500_skips_bad_records(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as fh:
            json.dump({"editions": [
                {"edition": "1993-06", "label": "June 1993",
                 "rmax_top": 59.7, "rmax_entry": 0.422,
                 "top_system": "x"},
                {"edition": "1994-06", "rmax_top": None, "rmax_entry": 1.0},
                {"rmax_top": 1.0, "rmax_entry": 1.0},
            ]}, fh)
            path = fh.name
        self.addCleanup(os.unlink, path)
        editions = hpl_np.load_top500(path)
        self.assertEqual([e["edition"] for e in editions], ["1993-06"])

    def test_report_without_data_file_is_silent(self):
        res = hpl_np.run_benchmark(200, seed=1, repeats=1)
        orig = hpl_np.load_top500
        hpl_np.load_top500 = lambda path=None: None
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                hpl_np.print_top500(res)
        finally:
            hpl_np.load_top500 = orig
        self.assertNotIn("TOP500", out.getvalue())

    def test_report_with_too_slow_machine(self):
        res = dict(hpl_np.run_benchmark(200, seed=1, repeats=1))
        res["gflops"] = 0.01
        editions = hpl_np.load_top500()
        if editions is None:
            self.skipTest("top500_data.json not next to hpl_np.py")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            hpl_np.print_top500(res)
        text = out.getvalue()
        self.assertIn("never have made ANY TOP500 list", text)
        self.assertIn("1993", text)

    def test_cli_no_top500_flag(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = hpl_np.main(["-n", "300", "--no-top500"])
        self.assertIn("PASSED", out.getvalue())
        self.assertEqual(rc, 0)
        self.assertNotIn("TOP500 calibration", out.getvalue())

    def test_cli_shows_calibration_with_bundled_data(self):
        if hpl_np.load_top500() is None:
            self.skipTest("top500_data.json not next to hpl_np.py")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = hpl_np.main(["-n", "300"])
        self.assertIn("TOP500 calibration", out.getvalue())
        self.assertIn("Gflop/s would have been", out.getvalue())
        self.assertEqual(rc, 0)


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


class TestSizing(unittest.TestCase):
    """The user-friendly size knob: GiB in, N out, RAM-based default."""

    def test_choose_size_explicit_n_wins(self):
        n, gib, ram = hpl_np.choose_size(n=100, gib=8)
        self.assertEqual(n, 100)
        self.assertAlmostEqual(gib, 100 * 100 * 8 / 2 ** 30)

    def test_choose_size_explicit_gib(self):
        n, gib, ram = hpl_np.choose_size(gib=8)
        self.assertEqual(n, 32768)          # the classic 8 GiB matrix
        self.assertEqual(gib, 8)

    def test_choose_size_default_is_25pct_of_ram(self):
        ram = 32 * 2 ** 30
        orig = hpl_np.physical_ram_bytes
        hpl_np.physical_ram_bytes = lambda: ram
        try:
            n, gib, ram_out = hpl_np.choose_size()
        finally:
            hpl_np.physical_ram_bytes = orig
        self.assertEqual(ram_out, ram)
        self.assertAlmostEqual(gib, 8.0)
        self.assertEqual(n, hpl_np.gib_to_n(8.0))

    def test_choose_size_fallback_without_ram_detection(self):
        orig = hpl_np.physical_ram_bytes
        hpl_np.physical_ram_bytes = lambda: None
        try:
            n, gib, ram_out = hpl_np.choose_size()
        finally:
            hpl_np.physical_ram_bytes = orig
        self.assertEqual(n, 4096)
        self.assertEqual(ram_out, None)
        self.assertAlmostEqual(gib, hpl_np.n_to_gib(4096))

    def test_gib_n_round_trip_is_conservative(self):
        """The matrix for the returned N never EXCEEDS the requested
        GiB (floor on the N side), and is never smaller than half of
        it (one order of N can't lose more than ~2x in area)."""
        for gib in (0.25, 1, 4, 8, 16):
            n = hpl_np.gib_to_n(gib)
            self.assertLessEqual(hpl_np.n_to_gib(n), gib)
            self.assertGreater(hpl_np.n_to_gib(n), gib / 2)


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

    def test_n_and_gib_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            hpl_np.main(["-n", "100", "-g", "1"])

    def test_seed_option_accepted(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = hpl_np.main(["-n", "60", "--seed", "123", "--repeats",
                               "1"])
        self.assertEqual(code, 0)
        self.assertIn("seed=123", buf.getvalue())

    def test_print_report_shows_gib_and_n(self):
        res = hpl_np.run_benchmark(n=80, seed=6, repeats=1)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            hpl_np.print_report(res)
        out = buf.getvalue()
        self.assertIn("N=80", out)
        self.assertIn("GiB", out)
        self.assertIn("LAPACK factors a copy of A", out)

    def test_default_sizing_uses_25pct_of_ram(self):
        """No -n and no -g: the default matrix is 25% of the detected
        RAM (here faked to 64 MiB, so the run stays instant)."""
        ram = 64 * 2 ** 20
        orig = hpl_np.physical_ram_bytes
        hpl_np.physical_ram_bytes = lambda: ram
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                code = hpl_np.main(["--no-top500"])
        finally:
            hpl_np.physical_ram_bytes = orig
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("25% of that", out)
        n = hpl_np.gib_to_n(0.25 * ram / 2 ** 30)
        self.assertIn("N={}".format(n), out)
        self.assertIn("25% of that: {:.2f} GiB -> N={}".format(
            hpl_np.n_to_gib(n), n), out)

    def test_cli_gib_flag_sets_size(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = hpl_np.main(["-g", "0.001", "--no-top500"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        n = hpl_np.gib_to_n(0.001)
        self.assertIn("N={}".format(n), out)
        self.assertIn("Sizing: {:.2f} GiB matrix -> N={}".format(
            hpl_np.n_to_gib(n), n), out)

    def test_run_prints_phases_and_progress(self):
        """No silence: every phase is announced, and each timed run
        reports its own Gflop/s as it completes."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = hpl_np.main(["-g", "0.001", "--repeats", "2"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("Phase 1/3", out)
        self.assertIn("Phase 2/3", out)
        self.assertIn("Phase 3/3", out)
        self.assertIn("warm-up", out)
        self.assertIn("run 1/2", out)
        self.assertIn("run 2/2", out)
        self.assertIn("Gflop/s", out)
        self.assertIn("best so far", out)

    def test_run_prints_try_other_sizes_tip(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = hpl_np.main(["-n", "60", "--repeats", "1"])
        self.assertEqual(code, 0)
        self.assertIn("-g 8", buf.getvalue())

    def test_no_top500_still_prints_tip(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = hpl_np.main(["-n", "60", "--no-top500"])
        self.assertEqual(code, 0)
        self.assertIn("-g 8", buf.getvalue())
        self.assertNotIn("TOP500 calibration", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
