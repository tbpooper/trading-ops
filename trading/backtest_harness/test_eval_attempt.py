from trading.backtest_harness.eval_attempt import simulate_eval_attempt_daily
from trading.prop.lucid_black_25k.risk_governor import Lucid25kRules


def test_pass_in_5_days_without_consistency_breach():
    rules = Lucid25kRules()

    # 250/day for 5 days => 1250; largest day 250 => ratio 0.2
    r = simulate_eval_attempt_daily([250, 250, 250, 250, 250], rules=rules)
    assert r.passed
    assert r.reason == "target_hit"
    assert r.days == 5


def test_consistency_breach_by_big_day():
    rules = Lucid25kRules()

    # Day1: +800 => total 800, largest 800 => ratio 1.0 > 0.6 -> breach
    r = simulate_eval_attempt_daily([800], rules=rules)
    assert not r.passed
    assert r.reason == "consistency_breach"


def test_mll_breach_on_close():
    rules = Lucid25kRules()

    # Highest close stays 25000, floor=24000. Close at 24000 => breach.
    r = simulate_eval_attempt_daily([0], day_closes=[24000], rules=rules)
    assert not r.passed
    assert r.reason == "mll_breach"


def test_floor_locks_at_25100_when_high_close_26100():
    rules = Lucid25kRules()

    # Pump close to 26100 -> floor locks at 25100. Then close 25100 breaches.
    r = simulate_eval_attempt_daily([1100, -1000], day_closes=[26100, 25100], rules=rules)
    assert not r.passed
    assert r.reason == "mll_breach"
