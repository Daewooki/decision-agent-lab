# tests/test_questions.py
from dalab.questions import Choice, Noul, Score, permute_choices, to_systemone

Q = {
    "cls": Choice("Which?", {"a": "A", "b": "B", "c": "C", "d": "D"}),
    "yes": Noul("Is it?"),
    "lvl": Score("How bad?", ("low", "mid", "high")),
}

def test_types():
    assert Q["cls"].type == "choice" and Q["yes"].type == "noul" and Q["lvl"].type == "score"

def test_to_systemone_wire_format():
    wire = to_systemone(Q)
    assert wire["cls"] == {"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": "B", "c": "C", "d": "D"}}
    assert wire["yes"] == {"type": "noul", "instructions": "Is it?"}
    assert wire["lvl"] == {"type": "score", "instructions": "How bad?", "criteria": ["low", "mid", "high"]}

def test_permute_none_keeps_order():
    qs, order = permute_choices(Q, None)
    assert list(qs["cls"].criteria) == ["a", "b", "c", "d"]
    assert order == {"cls": ["a", "b", "c", "d"]}

def test_permute_is_deterministic_and_changes_order():
    qs1, order1 = permute_choices(Q, 7)
    qs2, order2 = permute_choices(Q, 7)
    assert order1 == order2
    assert sorted(order1["cls"]) == ["a", "b", "c", "d"]
    assert list(qs1["cls"].criteria) == order1["cls"]
    assert qs1["cls"].criteria["c"] == "C"
    assert any(permute_choices(Q, s)[1]["cls"] != ["a", "b", "c", "d"] for s in range(5))
    assert qs1["yes"] is Q["yes"] and qs1["lvl"] is Q["lvl"]
