# tests/test_metrics.py
import math
from dalab.bench.metrics import accuracy, bootstrap_ci, brier, ece, macro_f1, mcnemar_exact

def test_accuracy_and_macro_f1():
    y = ["allow", "ask", "deny", "deny"]
    p = ["allow", "deny", "deny", "ask"]
    assert accuracy(y, p) == 0.5
    # allow: P=1,R=1,F1=1 ; ask: P=0,R=0,F1=0 ; deny: P=0.5,R=0.5,F1=0.5 -> macro 0.5
    assert abs(macro_f1(y, p, ["allow", "ask", "deny"]) - 0.5) < 1e-12

def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    vals = [1] * 80 + [0] * 20
    lo, hi = bootstrap_ci(vals, seed=1)
    assert lo < 0.8 < hi and 0.7 < lo and hi < 0.9
    assert bootstrap_ci(vals, seed=1) == (lo, hi)
    assert bootstrap_ci([1, 1, 1]) == (1.0, 1.0)

def test_mcnemar_exact_known_values():
    assert abs(mcnemar_exact(5, 0) - 0.0625) < 1e-12   # 2 * 0.5**5
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(9, 1) - 0.021484375) < 1e-9  # 2 * P(X<=1 | n=10)

def test_ece_and_brier():
    assert abs(ece([0.9] * 10, [1] * 9 + [0])) < 1e-12
    assert abs(ece([0.9] * 10, [1] * 10) - 0.1) < 1e-12
    assert abs(ece([0.2, 0.8], [0, 1]) - 0.2) < 1e-12
    assert abs(brier([0.9, 0.2], [1, 0]) - ((0.1 ** 2 + 0.2 ** 2) / 2)) < 1e-12
    assert brier([1.0, 0.0], [1, 0]) == 0.0
