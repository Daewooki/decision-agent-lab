# tests/test_decision.py
import math
import pytest
from dalab.decision import Answer, Decision, answer_with_confidence, concentration

def test_concentration_extremes():
    assert concentration({"a": 1.0, "b": 0.0, "c": 0.0}) == 1.0
    assert abs(concentration({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25})) < 1e-12
    assert concentration({"a": 1.0}) == 1.0

def test_concentration_rejects_empty_distribution():
    with pytest.raises(ValueError):
        concentration({})

def test_concentration_matches_solar_decide_reported_value():
    # observed 2026-10-20: Solar Decide reported confidence 0.992275 for these probabilities
    probs = {"local_write": 0.000709, "outward": 0.000296, "read_only": 0.998765, "unclear": 0.00023}
    assert abs(concentration(probs) - 0.992275) < 1e-3

def test_answer_with_confidence_reported_computed_or_none():
    reported = answer_with_confidence("choice", {"a": 0.5, "b": 0.5}, 0.9, choice="a")
    assert reported.confidence == 0.9 and reported.confidence_source == "reported" and reported.choice == "a"
    computed = answer_with_confidence("choice", {"a": 1.0, "b": 0.0}, None, choice="a")
    assert computed.confidence == 1.0 and computed.confidence_source == "computed"
    unknown = answer_with_confidence("choice", {}, None, choice="a")  # no distribution: nothing to be confident about
    assert unknown.confidence is None and unknown.confidence_source is None and unknown.probabilities is None

def test_decision_roundtrip():
    d = Decision(
        backend="typesafe", model_requested="jev-1.14.0", model_reported="jev-1.14.0",
        ts_utc="2026-10-20T00:00:00.000+00:00", location="Seoul, KR", latency_ms=312.5, server_latency_ms=None,
        answers={"cls": Answer("choice", choice="a", probabilities={"a": 0.9, "b": 0.1}, confidence=0.53, confidence_source="reported"),
                 "yes": Answer("noul", noul=0.2)},
        usage_input_tokens=400, usage_output_tokens=0, cost_usd=0.0000168, option_order={"cls": ["a", "b"]},
    )
    d2 = Decision.from_dict(d.to_dict())
    assert d2 == d
    assert d2.answers["cls"].choice == "a" and d2.answers["yes"].noul == 0.2
    assert d.error is None and d.raw == {} and d.extra == {}
