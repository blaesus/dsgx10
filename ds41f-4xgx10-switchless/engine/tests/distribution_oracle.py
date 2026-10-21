"""Independent exact-law oracle; no imports from the candidate engine.

Run with CPython only. The generated header supplies probability fixtures to the
actual ops::sample CUDA test; its expected law is calculated with Fractions,
not with the candidate's logsumexp, RNG, acceptance or residual code.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import unittest


def norm(values):
    values = tuple(F(x) for x in values)
    if any(x < 0 for x in values) or sum(values) <= 0:
        raise ValueError("invalid probability mass")
    total = sum(values)
    return tuple(x / total for x in values)


CASES = (
    ("equal", norm([1, 2, 3, 4]), norm([1, 2, 3, 4])),
    ("disjoint", norm([1, 2, 0, 0]), norm([0, 0, 3, 1])),
    ("zeros", norm([0, 2, 3, 0]), norm([1, 0, 3, 0])),
    ("tiny_target", norm([F(1, 2**30), 1, 2, 3]), norm([3, 2, 1, 0])),
    ("high_entropy", norm(range(1, 17)), norm(range(16, 0, -1))),
    ("target_point", norm([0, 0, 1, 0]), norm([1, 1, 1, 1])),
    ("proposal_point", norm([1, 2, 3, 4]), norm([0, 1, 0, 0])),
    ("same_point", norm([0, 0, 0, 1]), norm([0, 0, 0, 1])),
    ("near_equal", norm([1000001, 999999, 1000000, 1000000]),
     norm([1000000] * 4)),
)


def exact_law(p, q):
    if len(p) != len(q) or sum(p) != 1 or sum(q) != 1:
        raise ValueError("probabilities must be normalized and same size")
    accepted = tuple(min(a, b) for a, b in zip(p, q))
    rejection = 1 - sum(accepted)
    positive = tuple(max(a - b, F(0)) for a, b in zip(p, q))
    assert sum(positive) == rejection
    residual = tuple(x / rejection for x in positive) if rejection else None
    output = tuple(a + (rejection * residual[i] if residual else 0)
                   for i, a in enumerate(accepted))
    return accepted, rejection, residual, output


def truncate(p, top_k=0, top_p=F(1)):
    """Exact effective distribution; ties have deterministic lowest-ID order."""
    ids = sorted(range(len(p)), key=lambda i: (-p[i], i))
    if top_k > 0:
        ids = ids[:top_k]
    total = sum(p[i] for i in ids)
    if top_p < 1:
        keep, cumulative = [], F(0)
        for i in ids:
            keep.append(i)
            cumulative += p[i] / total
            if cumulative >= top_p:
                break
        ids = keep
    total = sum(p[i] for i in ids)
    return tuple(p[i] / total if i in ids else F(0) for i in range(len(p)))


def block_outputs(p_fn, q_fn, draft_depth=2):
    """Enumerate all proposal/reject/bonus paths, stop at first rejection."""
    result = defaultdict(F)

    def visit(prefix, mass, j):
        p = p_fn(prefix)
        if j == draft_depth:
            for token, prob in enumerate(p):
                result[prefix + (token,)] += mass * prob
            return
        q = q_fn(prefix)
        _, _, residual, _ = exact_law(p, q)
        for token, proposed in enumerate(q):
            if not proposed:
                continue
            accept = min(F(1), p[token] / proposed)
            visit(prefix + (token,), mass * proposed * accept, j + 1)
            rejected = mass * proposed * (1 - accept)
            if rejected:
                assert residual is not None
                for corrected, prob in enumerate(residual):
                    result[prefix + (corrected,)] += rejected * prob
    visit((), F(1), 0)
    return {tokens: mass for tokens, mass in result.items() if mass}


class ExactOracleTests(unittest.TestCase):
    def test_fixture_laws(self):
        for name, p, q in CASES:
            with self.subTest(name=name):
                accepted, rejection, residual, output = exact_law(p, q)
                self.assertEqual(output, p)
                self.assertEqual(sum(accepted) + rejection, 1)
                if residual is not None:
                    self.assertEqual(sum(residual), 1)
                    self.assertTrue(all(residual[i] == 0 for i in range(len(p))
                                        if p[i] <= q[i]))

    def test_all_small_rational_distributions(self):
        distributions = [norm(v) for v in itertools.product(range(4), repeat=3)
                         if sum(v)]
        for p, q in itertools.product(distributions, repeat=2):
            self.assertEqual(exact_law(p, q)[3], p)

    def test_greedy_proposal_is_valid(self):
        # The existing deterministic proposal is not a wrong target law.
        p = norm([1, 2, 3, 4])
        for d in range(4):
            q = tuple(F(i == d) for i in range(4))
            accepted, reject, residual, out = exact_law(p, q)
            self.assertEqual(accepted[d], p[d])
            self.assertEqual(reject, 1 - p[d])
            self.assertEqual(residual[d], 0)
            self.assertEqual(out, p)

    def test_effective_top_k_and_top_p(self):
        p, q = norm([4, 3, 2, 1]), norm([1, 2, 3, 4])
        for k, cutoff in itertools.product((0, 1, 2, 4), (F(1), F(1, 2), F(9, 10))):
            pp, qq = truncate(p, k, cutoff), truncate(q, k, cutoff)
            self.assertEqual(exact_law(pp, qq)[3], pp)
        self.assertEqual(truncate(norm([1, 1, 1]), 1), (1, 0, 0))

    def test_first_reject_and_bonus(self):
        p_fn = lambda prefix: norm([1 + sum(prefix), 3 + len(prefix)])
        q_fn = lambda prefix: norm([3 + len(prefix), 1 + sum(prefix)])
        out = block_outputs(p_fn, q_fn)
        self.assertEqual(sum(out.values()), 1)
        self.assertTrue(all(1 <= len(t) <= 3 for t in out))
        first = [sum(m for t, m in out.items() if t[0] == i) for i in range(2)]
        self.assertEqual(tuple(first), p_fn(()))
        # Complete every early-rejection path using target-only autoregression.
        full = defaultdict(F)
        def complete(tokens, mass):
            if len(tokens) == 3:
                full[tokens] += mass
            else:
                for tok, p in enumerate(p_fn(tokens)):
                    complete(tokens + (tok,), mass * p)
        for tokens, mass in out.items():
            complete(tokens, mass)
        for tokens, mass in full.items():
            expected = F(1)
            for j, tok in enumerate(tokens):
                expected *= p_fn(tokens[:j])[tok]
            self.assertEqual(mass, expected)

    def test_stop_and_output_cap_do_not_publish_later_rows(self):
        def publish(row, stop, cap):
            out = []
            for token in row:
                if len(out) >= cap:
                    break
                out.append(token)
                if token in stop:
                    break
            return out
        self.assertEqual(publish([3, 1, 7, 8], {1}, 8), [3, 1])
        self.assertEqual(publish([3, 7, 8], {1}, 2), [3, 7])
        self.assertEqual(publish([1, 7, 8], {1}, 8), [1])

    def test_looking_ahead_to_choose_draft_depth_can_bias(self):
        # Ratio gate may inspect survival[1], which includes draft[0], before
        # deciding whether draft[0] is verified at all. This is not a stopping
        # decision measurable before that proposal draw.
        p, q = norm([1, 3]), norm([1, 1])
        _, _, residual, _ = exact_law(p, q)
        costs = (F(1), F(8, 5), F(9, 5))
        included = []
        for draft in range(2):
            survival = (F(1, 2), F(9, 20) if draft == 0 else F(1, 20))
            rates = (1 / costs[0], (1 + survival[0]) / costs[1],
                     (1 + sum(survival)) / costs[2])
            included.append(max(range(3), key=lambda n: rates[n]) > 0)
        self.assertEqual(included, [True, False])
        output = [F(0), F(0)]
        for draft in range(2):
            if included[draft]:
                acceptance = min(F(1), p[draft] / q[draft])
                output[draft] += q[draft] * acceptance
                for i in range(2):
                    output[i] += q[draft] * (1 - acceptance) * residual[i]
            else:
                for i in range(2):
                    output[i] += q[draft] * p[i]
        self.assertEqual(tuple(output), (F(3, 8), F(5, 8)))
        self.assertNotEqual(tuple(output), p)


def freeze(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, p, q in CASES:
        accepted, rejection, residual, output = exact_law(p, q)
        records.append({
            "name": name, "p": [str(x) for x in p], "q": [str(x) for x in q],
            "accepted_mass": [str(x) for x in accepted],
            "acceptance_rate": str(1 - rejection),
            "residual": None if residual is None else [str(x) for x in residual],
            "output": [str(x) for x in output],
        })
    report = {
        "schema": 1, "oracle": "exact rational probability mass, candidate-independent",
        "cases": records,
        "limits": [
            "No GPU execution, engine integration or distribution implementation verified by this file.",
            "Finite 23-bit RNG cannot resolve arbitrarily small masses; tiny case checks exact law separately.",
            "CUDA empirical checks cannot prove all-distribution equivalence.",
        ],
        "references": [
            "https://proceedings.mlr.press/v202/leviathan23a.html",
            "https://arxiv.org/abs/2302.01318",
        ],
    }
    raw = (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode()
    (out_dir / "distribution-fixtures.json").write_bytes(raw)
    header = ["#pragma once", "#include <cstddef>",
              "struct OracleCase { const char* name; int n; double p[16], q[16], accept; };",
              "static const OracleCase kOracleCases[] = {"]
    for name, p, q in CASES:
        a = sum(min(x, y) for x, y in zip(p, q))
        fmt = lambda xs: ",".join(format(float(x), ".17g") for x in xs)
        header.append('{"%s",%d,{%s},{%s},%.17g},' %
                      (name, len(p), fmt(p), fmt(q), float(a)))
    header.extend(["};", "static constexpr std::size_t kOracleCaseCount = sizeof(kOracleCases)/sizeof(kOracleCases[0]);"])
    (out_dir / "distribution_fixtures.h").write_bytes(("\n".join(header) + "\n").encode("utf-8"))
    return hashlib.sha256(raw).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ExactOracleTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.freeze:
        print(json.dumps({"fixture_sha256": freeze(args.freeze),
                          "tests": result.testsRun, "passed": True}))
