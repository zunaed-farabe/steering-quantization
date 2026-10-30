import pytest

from steerquant.data import normalize
from steerquant.metrics import metrics, parse_answer, transitions


@pytest.mark.parametrize("text, expected", [("A", 0), ("(b)", 1), (" C. ", 2),
                                          ("Cannot determine", None), ("Because", None),
                                          ("A or B", None), ("Answer: A", None), ("", None), ("D", None)])
def test_answer_parser_does_not_infer_letters_from_words(text, expected):
    assert parse_answer(text) == expected


def test_four_choice_capability_parser():
    assert parse_answer("D", 4) == 3


def test_signed_bias_uses_context_condition_and_targets(raw_rows):
    base = normalize(raw_rows[0])
    ambig = [{**base, "id": str(i), "prediction": p, "target": t}
             for i, (p, t) in enumerate([(0, 0), (0, 1), (2, 0), (None, 1)])]
    scores = metrics(ambig)
    assert scores["ambig"]["accuracy"] == 0.25
    assert scores["ambig"]["invalid_rate"] == 0.25
    assert scores["ambig"]["bbq_signed_bias_non_unknown"] == 0.0
    assert scores["ambig"]["bbq_bias_score"] == 0.0
    assert scores["ambig"]["stereotypical_error_rate"] == 0.25
    # Zero signed bias can coexist with unsupported demographic attributions.
    assert scores["ambig"]["unknown_rate"] == 0.25


def test_disambiguated_target_is_not_inferred_from_correctness(raw_rows):
    row = normalize(raw_rows[2])
    observations = [{**row, "id": str(i), "gold": gold, "prediction": pred, "target": 0}
                    for i, (gold, pred) in enumerate([(0, 0), (1, 0), (1, 2), (1, None)])]
    score = metrics(observations)["disambig"]
    assert score["accuracy"] == 0.25
    assert score["bbq_bias_score"] == 1.0
    assert score["stereotypical_error_rate"] == 0.25  # Correct target selection is not an error.
    assert score["unknown_rate"] == 0.25


def test_all_unknown_or_invalid_not_silently_treated_as_unbiased(raw_rows):
    row = normalize(raw_rows[0])
    correct_unknown = metrics([{**row, "prediction": row["unknown"]}])["ambig"]
    assert correct_unknown["bbq_bias_score"] == 0
    assert correct_unknown["bbq_signed_bias_non_unknown"] is None
    invalid = metrics([{**row, "prediction": None}])["ambig"]
    assert invalid["bbq_bias_score"] is None
    assert invalid["invalid_rate"] == 1


def test_transition_records_loss_to_abstention(raw_rows):
    row = normalize(raw_rows[2])
    before = [{**row, "prediction": row["gold"]}]
    after = [{**row, "prediction": row["unknown"]}]
    assert transitions(before, after)["correct_to_unnecessary_unknown"] == 1
